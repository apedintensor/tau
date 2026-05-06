import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from r2 import (
    _is_public_task_leakage_key,
    publish_duel_data,
    publish_round_data,
    publish_training_data,
)
from workspace import build_compare_paths, build_solution_paths, build_task_paths


class FakeS3Client:
    def __init__(self):
        self.puts = []
        self.deletes = []

    def put_object(self, **kwargs):
        self.puts.append(kwargs)
        return {}

    def delete_object(self, **kwargs):
        self.deletes.append(kwargs)
        return {}


def _json_body(put):
    body = put["Body"]
    if isinstance(body, bytes):
        body = body.decode()
    return json.loads(body)


class R2PublicSanitizationTest(unittest.TestCase):
    def test_publish_round_data_keeps_requested_public_round_artifacts(self):
        client = FakeS3Client()
        with tempfile.TemporaryDirectory() as tmp:
            tasks_root = Path(tmp)
            task_paths = build_task_paths(tasks_root, "validate-1")
            task_paths.task_dir.mkdir(parents=True)
            task_paths.solutions_dir.mkdir()
            task_paths.comparisons_dir.mkdir()

            task_paths.task_json_path.write_text(
                json.dumps(
                    {
                        "repo_full_name": "source/repo",
                        "commit_sha": "target-sha",
                        "task": {"prompt_text": "private task prompt", "title": "private title"},
                    }
                )
            )
            task_paths.task_txt_path.write_text("private task prompt\n")
            task_paths.reference_patch_path.write_text("private reference patch\n")
            task_paths.commit_path.write_text(
                json.dumps({"commit_sha": "target-sha", "combined_patch": "private reference patch"})
            )

            for name in ("baseline", "king", "challenger"):
                sol_paths = build_solution_paths(task_paths, name)
                sol_paths.root.mkdir(parents=True)
                sol_paths.solution_diff_path.write_text(f"{name} public diff\n")
                sol_paths.rollout_jsonl_path.write_text('{"prompt":"private task prompt"}\n')
                sol_paths.solve_json_path.write_text(
                    json.dumps(
                        {
                            "stage": "solve",
                            "task_name": "validate-1",
                            "solution_name": name,
                            "repo_full_name": "source/repo",
                            "commit_sha": "target-sha",
                            "result": {
                                "raw_output": "private task prompt",
                                "rollout_format": "single-file-json",
                                "solution_diff": f"{name} public diff\n",
                                "session_id": "internal-session",
                                "rollout_filename": "rollout.jsonl",
                                "model": "solver/model",
                                "exit_reason": "completed",
                                "total_tokens": 123,
                            },
                        }
                    )
                )

            for cmp_name in ("king--vs--baseline", "challenger--vs--baseline", "king--vs--challenger"):
                cmp_paths = build_compare_paths(task_paths, cmp_name)
                cmp_paths.root.mkdir(parents=True)
                cmp_paths.compare_json_path.write_text(
                    json.dumps(
                        {
                            "repo_full_name": "source/repo",
                            "commit_sha": "target-sha",
                            "result": {"similarity_ratio": 0.5},
                        }
                    )
                )

            with patch("r2._get_s3_client", return_value=client):
                self.assertTrue(
                    publish_round_data(
                        duel_id=7,
                        task_name="validate-1",
                        tasks_root=tasks_root,
                    )
                )

        put_keys = {item["Key"] for item in client.puts}
        all_uploaded = "\n".join(
            (item["Body"].decode() if isinstance(item["Body"], bytes) else str(item["Body"]))
            for item in client.puts
        )

        self.assertNotIn("sn66/duels/000007/rounds/validate-1/task.txt", put_keys)
        self.assertNotIn("sn66/duels/000007/rounds/validate-1/task.json", put_keys)
        self.assertNotIn("sn66/duels/000007/rounds/validate-1/reference.patch", put_keys)
        self.assertNotIn("sn66/duels/000007/rounds/validate-1/commit.json", put_keys)
        self.assertIn("sn66/duels/000007/rounds/validate-1/solutions/king.diff", put_keys)
        self.assertIn("sn66/duels/000007/rounds/validate-1/solutions/challenger.diff", put_keys)
        self.assertIn("sn66/duels/000007/rounds/validate-1/solutions/king.solve.json", put_keys)
        self.assertIn("sn66/duels/000007/rounds/validate-1/solutions/challenger.solve.json", put_keys)
        self.assertIn("sn66/duels/000007/rounds/validate-1/comparisons/king--vs--reference.json", put_keys)
        self.assertIn("sn66/duels/000007/rounds/validate-1/comparisons/challenger--vs--reference.json", put_keys)
        self.assertIn("sn66/duels/000007/rounds/validate-1/comparisons/king--vs--challenger.json", put_keys)
        self.assertNotIn("sn66/duels/000007/rounds/validate-1/solutions/baseline.diff", put_keys)
        self.assertNotIn("sn66/duels/000007/rounds/validate-1/solutions/baseline.solve.json", put_keys)
        self.assertFalse(any(key.endswith(".rollout.jsonl.gz") for key in put_keys))
        self.assertNotIn("private task prompt", all_uploaded)
        self.assertNotIn("private reference patch", all_uploaded)
        self.assertIn("king public diff", all_uploaded)
        self.assertIn("challenger public diff", all_uploaded)
        self.assertNotIn("baseline public diff", all_uploaded)
        self.assertNotIn("target-sha", all_uploaded)

        solve_put = next(item for item in client.puts if item["Key"].endswith("/solutions/king.solve.json"))
        solve_payload = _json_body(solve_put)
        self.assertNotIn("repo_full_name", solve_payload)
        self.assertNotIn("commit_sha", solve_payload)
        self.assertNotIn("raw_output", solve_payload["result"])
        self.assertNotIn("rollout_format", solve_payload["result"])
        self.assertNotIn("solution_diff", solve_payload["result"])
        self.assertNotIn("session_id", solve_payload["result"])
        self.assertNotIn("rollout_filename", solve_payload["result"])
        self.assertEqual(solve_payload["result"]["model"], "solver/model")

        compare_put = next(
            item
            for item in client.puts
            if item["Key"].endswith("/comparisons/king--vs--reference.json")
        )
        compare_payload = _json_body(compare_put)
        self.assertNotIn("repo_full_name", compare_payload)
        self.assertNotIn("commit_sha", compare_payload)
        self.assertEqual(compare_payload["result"]["similarity_ratio"], 0.5)

        deleted_keys = {item["Key"] for item in client.deletes}
        self.assertIn("sn66/duels/000007/rounds/validate-1/task.txt", deleted_keys)
        self.assertIn("sn66/duels/000007/rounds/validate-1/reference.patch", deleted_keys)
        self.assertIn("sn66/duels/000007/rounds/validate-1/solutions/baseline.diff", deleted_keys)
        self.assertIn("sn66/duels/000007/rounds/validate-1/solutions/baseline.solve.json", deleted_keys)
        self.assertIn("sn66/duels/000007/rounds/validate-1/solutions/king.rollout.jsonl.gz", deleted_keys)

    def test_publish_duel_data_strips_private_round_fields(self):
        client = FakeS3Client()
        duel = {
            "duel_id": 9,
            "rounds": [
                {
                    "task_name": "validate-1",
                    "winner": "king",
                    "task_root": "/private/task/root",
                    "king_compare_root": "/private/king/compare",
                    "challenger_compare_root": "/private/challenger/compare",
                    "llm_judge_rationale": "private task prompt and reference details",
                    "king_score": 0.8,
                }
            ],
        }

        with patch("r2._get_s3_client", return_value=client):
            self.assertTrue(publish_duel_data(duel_id=9, duel_dict=duel))

        payload = _json_body(client.puts[0])
        round_payload = payload["rounds"][0]
        self.assertNotIn("task_root", round_payload)
        self.assertNotIn("king_compare_root", round_payload)
        self.assertNotIn("challenger_compare_root", round_payload)
        self.assertNotIn("llm_judge_rationale", round_payload)
        self.assertEqual(round_payload["king_score"], 0.8)

    def test_publish_training_data_deletes_legacy_public_file_without_uploading(self):
        client = FakeS3Client()

        with patch("r2._get_s3_client", return_value=client):
            self.assertFalse(
                publish_training_data(
                    duel_id=11,
                    duel_dict={"rounds": []},
                    tasks_root=Path("/unused"),
                )
            )

        self.assertEqual(client.puts, [])
        self.assertEqual(client.deletes[0]["Key"], "sn66/duels/000011/training.jsonl")

    def test_public_task_leakage_key_detection_covers_legacy_public_objects(self):
        self.assertTrue(_is_public_task_leakage_key("sn66/duels/000001/rounds/a/task.txt"))
        self.assertTrue(_is_public_task_leakage_key("sn66/duels/000001/rounds/a/reference.patch"))
        self.assertTrue(_is_public_task_leakage_key("sn66/duels/000001/rounds/a/commit.json"))
        self.assertTrue(_is_public_task_leakage_key("sn66/duels/000001/rounds/a/task.json"))
        self.assertTrue(_is_public_task_leakage_key("sn66/duels/000001/rounds/a/solutions/baseline.solve.json"))
        self.assertTrue(_is_public_task_leakage_key("sn66/duels/000001/rounds/a/solutions/baseline.diff"))
        self.assertTrue(_is_public_task_leakage_key("sn66/duels/000001/rounds/a/solutions/king.rollout.jsonl.gz"))
        self.assertTrue(_is_public_task_leakage_key("sn66/duels/000001/training.jsonl"))
        self.assertFalse(_is_public_task_leakage_key("sn66/duels/000001/rounds/a/solutions/king.solve.json"))
        self.assertFalse(_is_public_task_leakage_key("sn66/duels/000001/rounds/a/solutions/challenger.solve.json"))
        self.assertFalse(_is_public_task_leakage_key("sn66/duels/000001/rounds/a/solutions/king.diff"))
        self.assertFalse(_is_public_task_leakage_key("sn66/duels/000001/rounds/a/solutions/challenger.diff"))
        self.assertFalse(_is_public_task_leakage_key("sn66/dashboard.json"))


if __name__ == "__main__":
    unittest.main()
