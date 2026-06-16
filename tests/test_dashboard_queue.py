from __future__ import annotations

import json
from pathlib import Path

from dashboard_queue import augment_dashboard_payload, queue_from_validator_state


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_queue_from_validator_state_uses_state_json(tmp_path: Path) -> None:
    validate_root = tmp_path / "netuid-66"
    queued_hotkeys = [
        f"5QueuedHotkey{i:02d}111111111111111111111111111111111"
        for i in range(6)
    ]
    state_queue = [
        {
            "uid": 100 + i,
            "hotkey": hotkey,
            "repo_full_name": f"private-submission/state-sub-{i}",
            "accepted_at": f"2026-06-16T11:{10 + i:02d}:00+00:00",
            "source": "private",
        }
        for i, hotkey in enumerate(queued_hotkeys)
    ]
    _write_json(validate_root / "state.json", {"queue": state_queue})

    merged = queue_from_validator_state(
        status={"queue": []},
        validate_root=validate_root,
    )
    assert [item["hotkey"] for item in merged] == queued_hotkeys
    assert all(item.get("repo") for item in merged)


def test_queue_from_validator_state_falls_back_to_snapshot_when_state_empty(
    tmp_path: Path,
) -> None:
    validate_root = tmp_path / "netuid-66"
    _write_json(validate_root / "state.json", {"queue": []})
    snapshot_queue = [
        {
            "uid": 7,
            "hotkey": "5ExistingHotkey1111111111111111111111111111111",
            "repo": "private-submission",
            "accepted_at": "2026-06-15T09:00:00+00:00",
        }
    ]

    merged = queue_from_validator_state(
        status={"queue": snapshot_queue},
        validate_root=validate_root,
    )
    assert [item["hotkey"] for item in merged] == [
        "5ExistingHotkey1111111111111111111111111111111"
    ]


def test_queue_from_validator_state_ignores_published_snapshot_when_state_has_queue(
    tmp_path: Path,
) -> None:
    validate_root = tmp_path / "netuid-66"
    _write_json(
        validate_root / "state.json",
        {
            "queue": [
                {
                    "uid": 42,
                    "hotkey": "5LiveHotkey1111111111111111111111111111111111",
                    "repo_full_name": "private-submission/live-sub",
                    "accepted_at": "2026-06-16T12:00:00+00:00",
                    "source": "private",
                }
            ]
        },
    )

    merged = queue_from_validator_state(
        status={
            "queue": [
                {
                    "uid": 99,
                    "hotkey": "5StaleHotkey111111111111111111111111111111111",
                    "accepted_at": "2026-06-15T09:00:00+00:00",
                }
            ]
        },
        validate_root=validate_root,
    )
    assert [item["hotkey"] for item in merged] == [
        "5LiveHotkey1111111111111111111111111111111111"
    ]


def test_augment_dashboard_payload_updates_timestamp(tmp_path: Path) -> None:
    validate_root = tmp_path / "netuid-66"
    dashboard_path = validate_root / "dashboard_data.json"
    _write_json(
        validate_root / "state.json",
        {
            "queue": [
                {
                    "uid": 7,
                    "hotkey": "5ExistingHotkey1111111111111111111111111111111",
                    "repo_full_name": "private-submission/existing-sub",
                    "accepted_at": "2026-06-15T09:00:00+00:00",
                    "source": "private",
                }
            ]
        },
    )

    payload = {
        "updated_at": "2026-06-15T13:32:18+00:00",
        "status": {"queue": []},
    }

    augmented = augment_dashboard_payload(payload, dashboard_data_path=dashboard_path)

    assert augmented["updated_at"] != payload["updated_at"]
    assert len(augmented["status"]["queue"]) == 1
    assert augmented["status"]["queue"][0]["uid"] == 7


def test_queue_from_validator_state_filters_dueled_and_disqualified(tmp_path: Path) -> None:
    validate_root = tmp_path / "netuid-66"
    dueled_hotkey = "5DueledHotkey111111111111111111111111111111111"
    pending_hotkey = "5PendingHotkey1111111111111111111111111111111"
    dueled_commitment = (
        "private-submission:sub-dueled:" + ("a" * 64)
    )
    _write_json(
        validate_root / "state.json",
        {
            "queue": [
                {
                    "uid": 19,
                    "hotkey": dueled_hotkey,
                    "commitment": dueled_commitment,
                    "repo_full_name": "private-submission/sub-dueled",
                    "accepted_at": "2026-06-16T10:39:50+00:00",
                    "source": "private",
                },
                {
                    "uid": 78,
                    "hotkey": pending_hotkey,
                    "commitment": "private-submission:sub-pending:" + ("b" * 64),
                    "repo_full_name": "private-submission/sub-pending",
                    "accepted_at": "2026-06-16T12:33:25+00:00",
                    "source": "private",
                },
            ],
            "disqualified_hotkeys": [],
            "dueled_challenger_commitments": {
                dueled_hotkey: [dueled_commitment],
            },
        },
    )

    merged = queue_from_validator_state(
        status={"queue": []},
        validate_root=validate_root,
    )

    assert [item["hotkey"] for item in merged] == [pending_hotkey]
