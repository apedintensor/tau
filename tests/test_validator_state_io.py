import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from private_submission import record_private_submission_acceptance
from validate import ValidatorState, _load_state, _merge_queued_submissions_from_disk_state, _save_state
from validator_state_io import (
    enqueue_private_submission_in_state,
    private_submission_validator_queue_entry,
)


class ValidatorStateIoTest(unittest.TestCase):
    def test_enqueue_appends_to_state_json(self) -> None:
        with TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            submission = private_submission_validator_queue_entry(
                hotkey="hk1",
                submission_id="sub-1",
                agent_sha256="abc123",
                registration_block=100,
                uid=7,
            )
            self.assertTrue(
                enqueue_private_submission_in_state(
                    state_path=state_path,
                    submission=submission,
                )
            )
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["queue"]), 1)
            self.assertEqual(payload["queue"][0]["commitment"], submission["commitment"])
            self.assertEqual(payload["locked_commitments"]["hk1"], submission["commitment"])

    def test_enqueue_is_idempotent(self) -> None:
        with TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            submission = private_submission_validator_queue_entry(
                hotkey="hk1",
                submission_id="sub-1",
                agent_sha256="abc123",
                registration_block=100,
                uid=7,
            )
            self.assertTrue(
                enqueue_private_submission_in_state(state_path=state_path, submission=submission)
            )
            self.assertFalse(
                enqueue_private_submission_in_state(state_path=state_path, submission=submission)
            )

    def test_record_acceptance_enqueues_when_state_path_provided(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "private-submissions"
            state_path = Path(tmp) / "state.json"
            record_private_submission_acceptance(
                root=root,
                hotkey="hk1",
                submission_id="sub-1",
                agent_sha256="abc123",
                registration_block=100,
                uid=7,
                validator_state_path=state_path,
            )
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["queue"]), 1)

    def test_save_state_merges_disk_queue_without_dropping_memory(self) -> None:
        with TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            disk_submission = private_submission_validator_queue_entry(
                hotkey="hk-disk",
                submission_id="sub-disk",
                agent_sha256="deadbeef",
                registration_block=200,
                uid=3,
            )
            enqueue_private_submission_in_state(
                state_path=state_path,
                submission=disk_submission,
            )
            memory = ValidatorState()
            memory_submission = private_submission_validator_queue_entry(
                hotkey="hk-mem",
                submission_id="sub-mem",
                agent_sha256="cafebabe",
                registration_block=201,
                uid=4,
            )
            from validate import ValidatorSubmission, _record_commitment_acceptance

            memory.queue.append(ValidatorSubmission.from_dict(memory_submission))
            _record_commitment_acceptance(memory, memory.queue[0])
            _save_state(state_path, memory)
            saved = _load_state(state_path)
            commitments = {submission.commitment for submission in saved.queue}
            self.assertIn(disk_submission["commitment"], commitments)
            self.assertIn(memory_submission["commitment"], commitments)

    def test_merge_queued_submissions_from_disk_state(self) -> None:
        disk_entry = private_submission_validator_queue_entry(
            hotkey="hk-disk",
            submission_id="sub-disk",
            agent_sha256="deadbeef",
            registration_block=200,
            uid=3,
        )
        disk = ValidatorState.from_dict({"queue": [disk_entry]})
        memory = ValidatorState()
        added = _merge_queued_submissions_from_disk_state(memory, disk)
        self.assertEqual(added, 1)
        self.assertEqual(memory.queue[0].commitment, disk_entry["commitment"])


if __name__ == "__main__":
    unittest.main()
