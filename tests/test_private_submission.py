import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import RunConfig
from private_submission import (
    private_submission_check_passed,
    private_submission_signature_payload,
    private_submission_registration_check,
    record_private_submission_acceptance,
    run_agent_smoke_checks,
    run_private_submission_checks,
    write_private_submission_bundle,
)
from validate import (
    ValidatorSubmission,
    ValidatorState,
    _build_agent_config,
    _fetch_chain_submissions,
    _is_private_submission,
    _refresh_queue,
    _submission_is_eligible,
)


HOTKEY = "5F3sa2TJAWMqDhXG6jhV4N8ko9SxwGy8TpaNS1repoTitleHkey"
SIGNATURE = "signed-by-hotkey"
BASE_AGENT = """\
from typing import Optional

def solve(repo_path: str, issue: str, model: Optional[str] = None, api_base: Optional[str] = None, api_key: Optional[str] = None):
    return {"patch": "", "logs": "", "steps": 0, "cost": None, "success": True}
"""
GOOD_AGENT = """\
from typing import Optional

def solve(repo_path: str, issue: str, model: Optional[str] = None, api_base: Optional[str] = None, api_key: Optional[str] = None):
    logs = "private submission"
    return {"patch": "", "logs": logs, "steps": 1, "cost": None, "success": True}
"""
BAD_AGENT = """\
import requests

def solve(repo_path, issue, model=None, api_base=None, api_key=None):
    return {"patch": "", "logs": "", "steps": 0, "cost": None, "success": True}
"""


class FakeResponse:
    status_code = 404

    def json(self):
        return {}


class FakeGithubClient:
    def get(self, path, params=None):
        return FakeResponse()


class FakeCommitments:
    def __init__(self, commitment: str):
        self.commitment = commitment

    def get_all_revealed_commitments(self, netuid):
        return {}

    def get_all_commitments(self, netuid):
        return {HOTKEY: self.commitment}

    def get_commitment_metadata(self, netuid, hotkey):
        return {"block": 123}


class FakeSubnets:
    def get_uid_for_hotkey_on_subnet(self, hotkey, netuid):
        return 42 if hotkey == HOTKEY else None


class FakeSubtensor:
    block = 456

    def __init__(self, commitment: str):
        self.commitments = FakeCommitments(commitment)
        self.subnets = FakeSubnets()
        self.substrate = None


def fake_signature_verifier(hotkey, payload, signature):
    expected = private_submission_signature_payload(
        hotkey=HOTKEY,
        submission_id="sub-1",
        agent_sha256=hashlib.sha256(GOOD_AGENT.encode("utf-8")).hexdigest(),
    )
    return hotkey == HOTKEY and payload == expected and signature == SIGNATURE


class PrivateSubmissionChecksTest(unittest.TestCase):
    def test_scope_guard_failure_skips_judge_and_rejects(self):
        result = run_private_submission_checks(
            hotkey=HOTKEY,
            submitted_agent_py=BAD_AGENT,
            base_agent_py=BASE_AGENT,
            openrouter_judge=lambda payload: {"verdict": "pass", "overall_score": 99},
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.checks["scope_guard"].status, "failed")
        self.assertEqual(result.checks["openrouter_judge"].status, "skipped")
        self.assertTrue(
            any("non-stdlib module `requests`" in item for item in result.checks["scope_guard"].findings)
        )

    def test_smoke_check_rejects_pyflakes_regressions(self):
        result = run_agent_smoke_checks(agent_py=GOOD_AGENT + "\nimport os\n")

        self.assertEqual(result.status, "failed")
        self.assertTrue(any("os" in item for item in result.findings))

    def test_scope_guard_rejects_validator_owned_contract_edits(self):
        changed_agent = GOOD_AGENT.replace(
            "def solve(repo_path: str, issue: str, model: Optional[str] = None, api_base: Optional[str] = None, api_key: Optional[str] = None):",
            "def solve(repo_path: str, issue: str, temperature: float = 0.1, model: Optional[str] = None, api_base: Optional[str] = None, api_key: Optional[str] = None):",
        )
        result = run_private_submission_checks(
            hotkey=HOTKEY,
            submitted_agent_py=changed_agent,
            base_agent_py=GOOD_AGENT,
            openrouter_judge=lambda payload: {"verdict": "pass", "overall_score": 90},
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.checks["scope_guard"].status, "failed")
        self.assertTrue(any("solve()" in item or "validator-owned" in item for item in result.checks["scope_guard"].findings))

    def test_passing_checks_can_be_written_as_bundle(self):
        result = run_private_submission_checks(
            hotkey=HOTKEY,
            submitted_agent_py=GOOD_AGENT,
            base_agent_py=BASE_AGENT,
            openrouter_judge=lambda payload: {
                "verdict": "pass",
                "overall_score": 88,
                "summary": "Looks like a real local submission.",
                "reasons": ["changes runtime logs"],
            },
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.checks["scope_guard"].status, "passed")
        self.assertEqual(result.checks["openrouter_judge"].status, "passed")

        with tempfile.TemporaryDirectory() as tmp:
            bundle = write_private_submission_bundle(
                root=Path(tmp),
                submission_id="sub-1",
                hotkey=HOTKEY,
                agent_py=GOOD_AGENT,
                check_result=result,
                signature=SIGNATURE,
            )

            self.assertTrue((bundle / "agent.py").is_file())
            self.assertTrue((bundle / "check_result.json").is_file())
            saved = json.loads((bundle / "check_result.json").read_text())
            self.assertEqual(saved["checks"]["openrouter_judge"]["metadata"]["judgment"]["overall_score"], 88)
            self.assertEqual(saved["ci_checks"]["openrouter_judge"]["score"], 88)
            self.assertEqual(saved["llm_judge"]["metadata"]["judgment"]["summary"], "Looks like a real local submission.")

    def test_registration_gate_allows_one_submission_per_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = private_submission_registration_check(
                root=root,
                hotkey=HOTKEY,
                submission_id="sub-1",
                agent_sha256="a" * 64,
                registration_block=100,
            )
            self.assertEqual(first.status, "passed")

            record_private_submission_acceptance(
                root=root,
                hotkey=HOTKEY,
                submission_id="sub-1",
                agent_sha256="a" * 64,
                registration_block=100,
            )
            second = private_submission_registration_check(
                root=root,
                hotkey=HOTKEY,
                submission_id="sub-2",
                agent_sha256="b" * 64,
                registration_block=100,
            )
            self.assertEqual(second.status, "failed")
            self.assertTrue(any("re-register" in finding for finding in second.findings))

            after_reregistration = private_submission_registration_check(
                root=root,
                hotkey=HOTKEY,
                submission_id="sub-2",
                agent_sha256="b" * 64,
                registration_block=101,
            )
            self.assertEqual(after_reregistration.status, "passed")

    def test_registration_gate_requires_registration_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = private_submission_registration_check(
                root=Path(tmp),
                hotkey=HOTKEY,
                submission_id="sub-1",
                agent_sha256="a" * 64,
                registration_block=None,
            )

        self.assertEqual(result.status, "failed")
        self.assertTrue(any("Registration block is required" in item for item in result.findings))

    def test_bundle_requires_matching_hotkey_signature(self):
        result = run_private_submission_checks(
            hotkey=HOTKEY,
            submitted_agent_py=GOOD_AGENT,
            base_agent_py=BASE_AGENT,
            openrouter_judge=lambda payload: {"verdict": "pass", "overall_score": 90},
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_private_submission_bundle(
                root=root,
                submission_id="sub-1",
                hotkey=HOTKEY,
                agent_py=GOOD_AGENT,
                check_result=result,
                signature=SIGNATURE,
            )

            self.assertTrue(
                private_submission_check_passed(
                    root,
                    "sub-1",
                    result.agent_sha256,
                    hotkey=HOTKEY,
                    signature_verifier=fake_signature_verifier,
                )
            )
            self.assertFalse(
                private_submission_check_passed(
                    root,
                    "sub-1",
                    result.agent_sha256,
                    hotkey="5GspoofedHotkeyCannotClaimThisSubmission",
                    signature_verifier=fake_signature_verifier,
                )
            )
            self.assertFalse(
                private_submission_check_passed(
                    root,
                    "sub-1",
                    result.agent_sha256,
                    hotkey=HOTKEY,
                    signature_verifier=lambda hotkey, payload, signature: False,
                )
            )


class PrivateSubmissionValidatorTest(unittest.TestCase):
    def test_fetch_chain_submissions_accepts_checked_private_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_private_submission_checks(
                hotkey=HOTKEY,
                submitted_agent_py=GOOD_AGENT,
                base_agent_py=BASE_AGENT,
                openrouter_judge=lambda payload: {"verdict": "pass", "overall_score": 90},
            )
            write_private_submission_bundle(
                root=root,
                submission_id="sub-1",
                hotkey=HOTKEY,
                agent_py=GOOD_AGENT,
                check_result=result,
                signature=SIGNATURE,
            )
            commitment = f"private-submission:sub-1:{result.agent_sha256}"
            config = RunConfig(
                validate_private_submission_watch=True,
                validate_private_submission_root=root,
                validate_hotkey_spent_since_block=None,
            )

            with patch("validate._verify_hotkey_signature", fake_signature_verifier):
                submissions = _fetch_chain_submissions(
                    subtensor=FakeSubtensor(commitment),
                    github_client=FakeGithubClient(),
                    config=config,
                )

                self.assertEqual(len(submissions), 1)
                self.assertEqual(submissions[0].source, "private")
                self.assertEqual(submissions[0].commit_sha, result.agent_sha256)
                self.assertTrue(
                    _submission_is_eligible(
                        subtensor=FakeSubtensor(commitment),
                        github_client=FakeGithubClient(),
                        config=config,
                        submission=submissions[0],
                    )
                )
                agent_config = _build_agent_config(config, submissions[0])
                self.assertEqual(agent_config.solver_agent_source.kind, "local_file")
                self.assertEqual(Path(agent_config.solver_agent_source.local_path).read_text(), GOOD_AGENT)

    def test_github_pr_commitments_are_not_submission_method(self):
        config = RunConfig(
            validate_hotkey_spent_since_block=None,
        )

        submissions = _fetch_chain_submissions(
            subtensor=FakeSubtensor(f"github-pr:unarbos/ninja#7@{'a' * 40}"),
            github_client=FakeGithubClient(),
            config=config,
        )

        self.assertEqual(submissions, [])

    def test_refresh_queue_spends_hotkey_only_after_checked_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            passing = run_private_submission_checks(
                hotkey=HOTKEY,
                submitted_agent_py=GOOD_AGENT,
                base_agent_py=BASE_AGENT,
                openrouter_judge=lambda payload: {"verdict": "pass", "overall_score": 90},
            )
            write_private_submission_bundle(
                root=root,
                submission_id="sub-1",
                hotkey=HOTKEY,
                agent_py=GOOD_AGENT,
                check_result=passing,
                signature=SIGNATURE,
            )
            failed_sha = hashlib.sha256(BAD_AGENT.encode("utf-8")).hexdigest()
            config = RunConfig(
                validate_private_submission_watch=True,
                validate_private_submission_root=root,
                validate_hotkey_spent_since_block=None,
            )

            with patch("validate._verify_hotkey_signature", fake_signature_verifier):
                rejected = _fetch_chain_submissions(
                    subtensor=FakeSubtensor(f"private-submission:missing:{failed_sha}"),
                    github_client=FakeGithubClient(),
                    config=config,
                )
                self.assertEqual(rejected, [])

                accepted = _fetch_chain_submissions(
                    subtensor=FakeSubtensor(f"private-submission:sub-1:{passing.agent_sha256}"),
                    github_client=FakeGithubClient(),
                    config=config,
                )
            state = ValidatorState()
            _refresh_queue(chain_submissions=accepted, config=config, state=state, subtensor=FakeSubtensor(""))

            self.assertEqual([item.hotkey for item in state.queue], [HOTKEY])
            self.assertEqual(state.locked_commitments[HOTKEY], f"private-submission:sub-1:{passing.agent_sha256}")

    def test_published_private_submission_is_no_longer_runtime_private(self):
        submission = ValidatorSubmission(
            hotkey=HOTKEY,
            uid=42,
            repo_full_name="unarbos/ninja",
            repo_url="https://github.com/unarbos/ninja.git",
            commit_sha="f" * 40,
            commitment=f"private-submission:sub-1:{'a' * 64}",
            commitment_block=123,
            source="private_published",
        )

        self.assertFalse(_is_private_submission(submission))


if __name__ == "__main__":
    unittest.main()
