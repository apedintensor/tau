import base64
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from config import RunConfig
from validate import (
    _BURN_KING_HOTKEY,
    _BURN_KING_UID,
    ValidatorState,
    ValidatorSubmission,
    _COMMITMENT_COOLDOWN_BLOCKS,
    _build_burn_king,
    _ensure_king,
    _fetch_chain_submissions,
    _github_pr_required_checks_passed,
    _is_burn_king,
    _maybe_set_weights,
    _merge_promoted_github_pr,
    _refresh_queue,
    _submission_is_eligible,
)


SHA = "a" * 40
BASE_SHA = "b" * 40
RESOLVED_SHA = "c" * 40
MERGE_SHA = "d" * 40
ANCESTOR_SHA = "e" * 40
MINER_HOTKEY = "5F3sa2TJAWMqDhXG6jhV4N8ko9SxwGy8TpaNS1repoTitleHkey"
PR_COMMITMENT = f"github-pr:unarbos/ninja#7@{SHA}"


class FakeResponse:
    def __init__(self, status_code, payload, *, text: str | None = None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else str(payload)

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
        if path == "/repos/unarbos/ninja/branches/main":
            return FakeResponse(200, {"commit": {"sha": BASE_SHA}})
        raise AssertionError(f"unexpected GitHub path: {path}")

    def put(self, path, json=None):
        raise AssertionError(f"unexpected GitHub PUT path: {path}")

    def post(self, path, json=None):
        raise AssertionError(f"unexpected GitHub POST path: {path}")

    def delete(self, path):
        raise AssertionError(f"unexpected GitHub DELETE path: {path}")


class ConflictResolvingGithubClient(FakeGithubClient):
    def __init__(self):
        super().__init__()
        self.head_sha = SHA
        self.merge_attempts = 0
        self.temp_merge_attempts = 0
        self.updates = []
        self.created_refs = []
        self.deleted_refs = []
        self.temp_branch = None

    def get(self, path, params=None):
        if path == "/repos/unarbos/ninja/pulls/7":
            return FakeResponse(
                200,
                {
                    "number": 7,
                    "state": "open",
                    "draft": False,
                    "title": f"{MINER_HOTKEY} Improve harness",
                    "html_url": "https://github.com/unarbos/ninja/pull/7",
                    "base": {
                        "ref": "main",
                        "sha": BASE_SHA,
                        "repo": {"full_name": "unarbos/ninja"},
                    },
                    "head": {
                        "ref": "feature/conflict",
                        "sha": self.head_sha,
                        "repo": {
                            "full_name": "miner/ninja",
                            "clone_url": "https://github.com/miner/ninja.git",
                        },
                    },
                },
            )
        if path == f"/repos/unarbos/ninja/compare/{BASE_SHA}...{SHA}":
            return FakeResponse(200, {"merge_base_commit": {"sha": ANCESTOR_SHA}})
        if path == "/repos/unarbos/ninja/contents/agent.py":
            ref = (params or {}).get("ref")
            if ref == BASE_SHA:
                return _github_content("def solve(repo_path, issue):\n    return {'patch': ''}\n", "baseblob")
            if ref == ANCESTOR_SHA:
                return _github_content("def solve(repo_path, issue):\n    return {}\n", "ancestorblob")
        if path == "/repos/miner/ninja/contents/agent.py":
            ref = (params or {}).get("ref")
            if ref == SHA:
                return _github_content("def solve(repo_path, issue):\n    return {'patch': 'winner'}\n", "headblob")
        return super().get(path, params=params)

    def put(self, path, json=None):
        if path == "/repos/unarbos/ninja/pulls/7/merge":
            self.merge_attempts += 1
            if self.merge_attempts != 1:
                raise AssertionError("original PR branch must not be retried after conflict resolution")
            return FakeResponse(
                405,
                {"message": "Pull Request has merge conflicts"},
                text='{"message":"Pull Request has merge conflicts","status":"405"}',
            )
        if path == "/repos/unarbos/ninja/contents/agent.py":
            if self.temp_branch is None:
                raise AssertionError("content update happened before temp branch creation")
            content = base64.b64decode(json["content"]).decode("utf-8")
            self.updates.append({"payload": json, "content": content})
            if json["branch"] != self.temp_branch:
                raise AssertionError(f"expected update on temp branch {self.temp_branch}, got {json['branch']}")
            if json["sha"] != "baseblob":
                raise AssertionError(f"expected base blob sha, got {json['sha']}")
            return FakeResponse(200, {"commit": {"sha": RESOLVED_SHA}})
        raise AssertionError(f"unexpected GitHub PUT path: {path}")

    def post(self, path, json=None):
        if path == "/repos/unarbos/ninja/git/refs":
            ref = json["ref"]
            if not ref.startswith(f"refs/heads/validator/resolve-pr-7-{SHA[:12]}-"):
                raise AssertionError(f"unexpected temp ref {ref}")
            if json["sha"] != BASE_SHA:
                raise AssertionError(f"expected temp branch from base head {BASE_SHA}, got {json['sha']}")
            self.temp_branch = ref.removeprefix("refs/heads/")
            self.created_refs.append(json)
            return FakeResponse(201, {"ref": ref, "object": {"sha": BASE_SHA}})
        if path == "/repos/unarbos/ninja/merges":
            self.temp_merge_attempts += 1
            if json["base"] != "main":
                raise AssertionError(f"expected base main, got {json['base']}")
            if json["head"] != self.temp_branch:
                raise AssertionError(f"expected temp branch head {self.temp_branch}, got {json['head']}")
            return FakeResponse(201, {"sha": MERGE_SHA})
        raise AssertionError(f"unexpected GitHub POST path: {path}")

    def delete(self, path):
        if path.startswith("/repos/unarbos/ninja/git/refs/heads/validator/resolve-pr-7-"):
            self.deleted_refs.append(path)
            return FakeResponse(204, {})
        raise AssertionError(f"unexpected GitHub DELETE path: {path}")


def _github_content(text: str, sha: str) -> FakeResponse:
    return FakeResponse(
        200,
        {
            "sha": sha,
            "encoding": "base64",
            "content": base64.b64encode(text.encode("utf-8")).decode("ascii"),
        },
    )


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


class FakeWeightSubtensor:
    def __init__(self):
        self.neurons = SimpleNamespace(
            neurons_lite=lambda netuid: [
                SimpleNamespace(uid=0),
                SimpleNamespace(uid=42),
            ],
        )
        self.subnets = SimpleNamespace(
            get_uid_for_hotkey_on_subnet=self._unexpected_hotkey_lookup,
        )
        self.extrinsics = SimpleNamespace(set_weights=self._set_weights)
        self.calls = []

    def _unexpected_hotkey_lookup(self, hotkey, netuid):
        raise AssertionError("burn king weights must not resolve a hotkey")

    def _set_weights(self, **kwargs):
        self.calls.append(kwargs)
        return "ok"


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

    def test_empty_king_initializes_to_burn_uid_zero_without_consuming_queue(self):
        client = FakeGithubClient()
        config = RunConfig(validate_github_pr_repo="unarbos/ninja", validate_github_pr_base="main")
        state = ValidatorState(queue=[_submission(commitment=PR_COMMITMENT, sha=SHA, block=123)])

        _ensure_king(state=state, github_client=client, config=config)

        self.assertIsNotNone(state.current_king)
        assert state.current_king is not None
        self.assertTrue(_is_burn_king(state.current_king))
        self.assertEqual(state.current_king.uid, _BURN_KING_UID)
        self.assertEqual(state.current_king.hotkey, _BURN_KING_HOTKEY)
        self.assertEqual(state.current_king.repo_full_name, "unarbos/ninja")
        self.assertEqual(state.current_king.commit_sha, BASE_SHA)
        self.assertEqual(len(state.queue), 1)

    def test_burn_king_weights_target_uid_zero_directly(self):
        client = FakeGithubClient()
        config = RunConfig(
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_wallet_name="wallet",
            validate_wallet_hotkey="hotkey",
        )
        state = ValidatorState(current_king=_build_burn_king(github_client=client, config=config))
        subtensor = FakeWeightSubtensor()

        with patch("validate.bt.Wallet", return_value=object()):
            _maybe_set_weights(subtensor=subtensor, config=config, state=state, current_block=100)

        self.assertEqual(len(subtensor.calls), 1)
        self.assertEqual(subtensor.calls[0]["uids"], [0, 42])
        self.assertEqual(subtensor.calls[0]["weights"], [1.0, 0.0])
        self.assertEqual(state.last_weight_block, 100)

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

    def test_promoted_pr_merge_conflict_uses_llm_resolver_then_retries(self):
        client = ConflictResolvingGithubClient()
        config = RunConfig(
            openrouter_api_key="or-key",
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
        )
        resolved_agent = (
            "def solve(repo_path, issue, model=None, api_base=None, api_key=None):\n"
            "    return {'patch': 'resolved', 'logs': '', 'steps': 0, 'cost': None, 'success': True}\n"
        )

        with patch("validate.complete_text", return_value=f"<resolved_agent_py>\n{resolved_agent}</resolved_agent_py>") as complete:
            merged = _merge_promoted_github_pr(
                github_client=client,
                config=config,
                submission=_github_pr_submission(),
            )

        self.assertEqual(merged.source, "github_pr_merged")
        self.assertEqual(merged.repo_full_name, "unarbos/ninja")
        self.assertEqual(merged.commit_sha, MERGE_SHA)
        self.assertEqual(client.merge_attempts, 1)
        self.assertEqual(client.temp_merge_attempts, 1)
        self.assertEqual(len(client.created_refs), 1)
        self.assertEqual(len(client.deleted_refs), 1)
        self.assertEqual(len(client.updates), 1)
        self.assertEqual(client.updates[0]["content"], resolved_agent)
        self.assertEqual(client.updates[0]["payload"]["branch"], client.temp_branch)
        complete.assert_called_once()

    def test_promoted_pr_merge_conflict_rejects_invalid_llm_resolution(self):
        client = ConflictResolvingGithubClient()
        config = RunConfig(
            openrouter_api_key="or-key",
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
        )
        original = _github_pr_submission()

        with patch("validate.complete_text", return_value="<resolved_agent_py>\ndef nope(\n</resolved_agent_py>"):
            merged = _merge_promoted_github_pr(
                github_client=client,
                config=config,
                submission=original,
            )

        self.assertEqual(merged.source, "github_pr")
        self.assertEqual(merged.commit_sha, SHA)
        self.assertEqual(client.merge_attempts, 1)
        self.assertEqual(client.temp_merge_attempts, 0)
        self.assertEqual(client.created_refs, [])
        self.assertEqual(client.updates, [])


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


def _github_pr_submission() -> ValidatorSubmission:
    return ValidatorSubmission(
        hotkey=MINER_HOTKEY,
        uid=42,
        repo_full_name="miner/ninja",
        repo_url="https://github.com/miner/ninja.git",
        commit_sha=SHA,
        commitment=PR_COMMITMENT,
        commitment_block=123,
        source="github_pr",
        pr_number=7,
        pr_url="https://github.com/unarbos/ninja/pull/7",
        base_repo_full_name="unarbos/ninja",
        base_ref="main",
    )


if __name__ == "__main__":
    unittest.main()
