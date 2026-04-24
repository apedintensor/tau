from __future__ import annotations

import gzip
import json
import logging
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
import httpx

log = logging.getLogger("swe-eval.r2")

_R2_KEY_PREFIX = "sn66/"
_DASHBOARD_KEY = f"{_R2_KEY_PREFIX}dashboard.json"
_DUELS_PREFIX = f"{_R2_KEY_PREFIX}duels/"
_INDEX_KEY = f"{_DUELS_PREFIX}index.json"

_client_lock = threading.Lock()
_cached_client = None
_client_resolved = False

# Circuit breaker for 429 / SlowDown bursts. When the storage backend
# (Hippius) starts rate-limiting us, retrying immediately just makes the
# throttle worse and burns CPU + log volume. We track the last 429-style
# failure and skip non-essential uploads for _THROTTLE_BACKOFF_SECONDS
# afterwards. Dashboard publishes still go through (they're the user-visible
# heartbeat) but per-artifact uploads back off.
_THROTTLE_LOCK = threading.Lock()
_THROTTLE_UNTIL = 0.0
_THROTTLE_BACKOFF_SECONDS = 60.0
_THROTTLE_LOG_INTERVAL = 30.0
_LAST_THROTTLE_LOG = 0.0
_SUPPRESSED_SINCE_LOG = 0


def _is_throttle_error(exc: BaseException) -> bool:
    """True if the exception looks like S3 rate-limiting (429 / SlowDown)."""
    if isinstance(exc, ClientError):
        meta = exc.response.get("ResponseMetadata", {}) if hasattr(exc, "response") else {}
        if meta.get("HTTPStatusCode") == 429:
            return True
        code = (exc.response.get("Error", {}) or {}).get("Code", "") if hasattr(exc, "response") else ""
        if code in ("SlowDown", "Throttling", "ThrottlingException", "TooManyRequests"):
            return True
    msg = str(exc)
    return "429" in msg or "SlowDown" in msg or "TooManyRequests" in msg


def _note_throttle() -> None:
    """Record that we hit a rate limit; future calls within the backoff
    window will be suppressed. Logs at most once per _THROTTLE_LOG_INTERVAL
    seconds with a count of suppressed uploads."""
    global _THROTTLE_UNTIL, _LAST_THROTTLE_LOG, _SUPPRESSED_SINCE_LOG  # noqa: PLW0603
    now = time.monotonic()
    with _THROTTLE_LOCK:
        _THROTTLE_UNTIL = now + _THROTTLE_BACKOFF_SECONDS
        if now - _LAST_THROTTLE_LOG > _THROTTLE_LOG_INTERVAL:
            log.warning(
                "R2 backend rate-limited (429); backing off %.0fs (suppressed %d uploads since last warning)",
                _THROTTLE_BACKOFF_SECONDS, _SUPPRESSED_SINCE_LOG,
            )
            _LAST_THROTTLE_LOG = now
            _SUPPRESSED_SINCE_LOG = 0


def _is_throttled() -> bool:
    """True if we're currently in the throttle backoff window."""
    global _SUPPRESSED_SINCE_LOG  # noqa: PLW0603
    if time.monotonic() < _THROTTLE_UNTIL:
        with _THROTTLE_LOCK:
            _SUPPRESSED_SINCE_LOG += 1
        return True
    return False


def _get_s3_client():
    """Return a cached boto3 S3 client, or None if credentials are missing."""
    global _cached_client, _client_resolved  # noqa: PLW0603
    if _client_resolved:
        return _cached_client
    with _client_lock:
        if _client_resolved:
            return _cached_client
        endpoint = os.environ.get("R2_URL")
        access_key = os.environ.get("R2_ACCESS_KEY_ID")
        secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
        if not all([endpoint, access_key, secret_key]):
            _cached_client = None
        else:
            _cached_client = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name="decentralized",
                config=BotoConfig(
                    signature_version="s3v4",
                    s3={"addressing_style": "path"},
                    retries={"max_attempts": 1, "mode": "standard"},
                    connect_timeout=10,
                    read_timeout=30,
                ),
            )
        _client_resolved = True
        return _cached_client


def _get_bucket() -> str:
    return os.environ.get("R2_BUCKET_NAME", "constantinople")


def _upload_json(key: str, data: Any) -> bool:
    """Upload a JSON-serializable object to R2. Returns True on success.
    Raises on failure so callers can decide whether to bookkeeping-track
    the failure (e.g. note throttle, log)."""
    client = _get_s3_client()
    if client is None:
        return False
    body = json.dumps(data, indent=2)
    client.put_object(
        Bucket=_get_bucket(),
        Key=key,
        Body=body.encode(),
        ContentType="application/json",
    )
    return True


def _upload_text(key: str, content: str, content_type: str = "text/plain") -> bool:
    """Upload text content to R2. Returns True on success."""
    client = _get_s3_client()
    if client is None:
        return False
    client.put_object(
        Bucket=_get_bucket(),
        Key=key,
        Body=content.encode(),
        ContentType=content_type,
    )
    return True


def publish_dashboard_data(
    *,
    current_king: dict[str, Any] | None,
    duel_history: list[dict[str, Any]],
    status: dict[str, Any] | None = None,
) -> bool:
    """Serialize and upload dashboard.json to R2. Returns True on success."""
    if _get_s3_client() is None:
        log.warning("R2 credentials not configured; skipping dashboard publish")
        return False

    payload = {
        "updated_at": datetime.now(tz=UTC).isoformat(),
        "current_king": current_king,
        "duels": duel_history,
        "status": status,
    }
    try:
        _upload_json(_DASHBOARD_KEY, payload)
        log.info("Published dashboard data to r2://%s/%s (%d duels)", _get_bucket(), _DASHBOARD_KEY, len(duel_history))
        return True
    except Exception as exc:
        if _is_throttle_error(exc):
            _note_throttle()
            return False
        log.exception("Failed to publish dashboard data to R2")
        return False


def duel_to_summary(duel_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract the fields the dashboard needs from a full DuelResult dict."""
    king_before = duel_dict.get("king_before", {})
    challenger = duel_dict.get("challenger", {})
    rounds = duel_dict.get("rounds", [])

    scored_rounds = [r for r in rounds if r.get("error") is None]
    king_ratios = [r["king_similarity_ratio"] for r in scored_rounds if "king_similarity_ratio" in r]
    challenger_ratios = [r["challenger_similarity_ratio"] for r in scored_rounds if "challenger_similarity_ratio" in r]

    return {
        "duel_id": duel_dict.get("duel_id"),
        "started_at": duel_dict.get("started_at"),
        "finished_at": duel_dict.get("finished_at"),
        "king_uid": king_before.get("uid"),
        "king_hotkey": king_before.get("hotkey"),
        "king_repo": king_before.get("repo_full_name"),
        "king_repo_url": f"https://github.com/{king_before.get('repo_full_name', '')}",
        "king_commit_sha": king_before.get("commit_sha"),
        "king_commitment_block": king_before.get("commitment_block"),
        "challenger_uid": challenger.get("uid"),
        "challenger_hotkey": challenger.get("hotkey"),
        "challenger_repo": challenger.get("repo_full_name"),
        "challenger_repo_url": f"https://github.com/{challenger.get('repo_full_name', '')}",
        "challenger_commit_sha": challenger.get("commit_sha"),
        "challenger_commitment_block": challenger.get("commitment_block"),
        "king_similarity_ratio_mean": (sum(king_ratios) / len(king_ratios)) if king_ratios else 0.0,
        "challenger_similarity_ratio_mean": (sum(challenger_ratios) / len(challenger_ratios)) if challenger_ratios else 0.0,
        "wins": duel_dict.get("wins", 0),
        "losses": duel_dict.get("losses", 0),
        "ties": duel_dict.get("ties", 0),
        "king_replaced": duel_dict.get("king_replaced", False),
        "disqualification_reason": duel_dict.get("disqualification_reason"),
        "rounds": [
            {
                "task_name": r.get("task_name"),
                "winner": r.get("winner"),
                "king_similarity_ratio": r.get("king_similarity_ratio", 0.0),
                "challenger_similarity_ratio": r.get("challenger_similarity_ratio", 0.0),
                "king_challenger_similarity": r.get("king_challenger_similarity", 0.0),
                "king_lines": r.get("king_lines", 0),
                "challenger_lines": r.get("challenger_lines", 0),
                "baseline_lines": r.get("baseline_lines", 0),
            }
            for r in scored_rounds
        ],
    }


def _duel_key_prefix(duel_id: int) -> str:
    return f"{_DUELS_PREFIX}{duel_id:06d}/"


def _round_key_prefix(duel_id: int, task_name: str) -> str:
    return f"{_duel_key_prefix(duel_id)}rounds/{task_name}/"


def publish_round_data(
    *,
    duel_id: int,
    task_name: str,
    tasks_root: Path,
    solution_labels: dict[str, str] | None = None,
) -> bool:
    """Upload all artifacts for a single validation round to R2.

    Reads task.json, reference.patch, solution diffs, solve metadata,
    rollouts, and comparison JSONs from the local task workspace and
    uploads them under:
        sn66/duels/{duel_id}/rounds/{task_name}/...

    ``solution_labels`` maps canonical R2 names to actual on-disk solution
    folder names, e.g. ``{"baseline": "baseline", "challenger": "challenger-42"}``.
    When *None*, falls back to the canonical names as-is.

    Returns True if at least one file was uploaded, False otherwise.
    """
    if _get_s3_client() is None:
        return False
    if _is_throttled():
        return False

    from workspace import build_compare_paths, build_solution_paths, build_task_paths

    prefix = _round_key_prefix(duel_id, task_name)
    task_paths = build_task_paths(tasks_root, task_name)
    labels = solution_labels or {}
    uploaded = 0

    def _handle_upload_exc(exc: BaseException, r2_key: str) -> None:
        if _is_throttle_error(exc):
            _note_throttle()
            return
        log.exception("Failed to upload %s to R2 (non-fatal)", r2_key)

    def _try_upload_json_file(local_path: Path, r2_key: str) -> None:
        nonlocal uploaded
        if not local_path.exists() or _is_throttled():
            return
        try:
            data = json.loads(local_path.read_text())
            _upload_json(r2_key, data)
            uploaded += 1
        except Exception as exc:
            _handle_upload_exc(exc, r2_key)

    def _try_upload_text_file(local_path: Path, r2_key: str, content_type: str = "text/plain") -> None:
        nonlocal uploaded
        if not local_path.exists() or _is_throttled():
            return
        try:
            _upload_text(r2_key, local_path.read_text(), content_type)
            uploaded += 1
        except Exception as exc:
            _handle_upload_exc(exc, r2_key)

    def _try_upload_gzip_file(local_path: Path, r2_key: str) -> None:
        nonlocal uploaded
        if not local_path.exists() or _is_throttled():
            return
        try:
            raw = local_path.read_bytes()
            compressed = gzip.compress(raw)
            client = _get_s3_client()
            if client is None:
                return
            client.put_object(
                Bucket=_get_bucket(),
                Key=r2_key,
                Body=compressed,
                ContentType="application/x-ndjson",
                ContentEncoding="gzip",
            )
            uploaded += 1
        except Exception as exc:
            _handle_upload_exc(exc, r2_key)

    _try_upload_json_file(task_paths.task_json_path, f"{prefix}task.json")
    _try_upload_text_file(task_paths.task_txt_path, f"{prefix}task.txt")
    _try_upload_text_file(task_paths.reference_patch_path, f"{prefix}reference.patch", "text/x-diff")
    _try_upload_json_file(task_paths.commit_path, f"{prefix}commit.json")

    canonical_names = ("baseline", "king", "challenger")
    for canonical in canonical_names:
        disk_name = labels.get(canonical, canonical)
        sol_paths = build_solution_paths(task_paths, disk_name)
        _try_upload_text_file(
            sol_paths.solution_diff_path,
            f"{prefix}solutions/{canonical}.diff",
            "text/x-diff",
        )
        _try_upload_json_file(
            sol_paths.solve_json_path,
            f"{prefix}solutions/{canonical}.solve.json",
        )
        _try_upload_gzip_file(
            sol_paths.rollout_jsonl_path,
            f"{prefix}solutions/{canonical}.rollout.jsonl.gz",
        )

    compare_pairs = [
        ("baseline", "king"),
        ("baseline", "challenger"),
        ("king", "challenger"),
    ]
    for left_canonical, right_canonical in compare_pairs:
        left_disk = labels.get(left_canonical, left_canonical)
        right_disk = labels.get(right_canonical, right_canonical)
        disk_cmp_name = f"{left_disk}--vs--{right_disk}"
        r2_cmp_name = f"{left_canonical}--vs--{right_canonical}"
        cmp_paths = build_compare_paths(task_paths, disk_cmp_name)
        _try_upload_json_file(
            cmp_paths.compare_json_path,
            f"{prefix}comparisons/{r2_cmp_name}.json",
        )

    log.info(
        "Published %d round artifacts for duel %d task %s to R2",
        uploaded, duel_id, task_name,
    )
    return uploaded > 0


def publish_duel_data(*, duel_id: int, duel_dict: dict[str, Any]) -> bool:
    """Upload the full DuelResult JSON to R2.

    Writes to: sn66/duels/{duel_id}/duel.json
    """
    if _get_s3_client() is None:
        return False
    if _is_throttled():
        return False
    key = f"{_duel_key_prefix(duel_id)}duel.json"
    try:
        _upload_json(key, duel_dict)
        log.info("Published duel %d to r2://%s/%s", duel_id, _get_bucket(), key)
        return True
    except Exception as exc:
        if _is_throttle_error(exc):
            _note_throttle()
            return False
        log.exception("Failed to publish duel %d to R2 (non-fatal)", duel_id)
        return False


def publish_duel_index(
    *,
    duel_history: list[dict[str, Any]],
    latest_duel_dict: dict[str, Any] | None = None,
) -> bool:
    """Rebuild and upload sn66/duels/index.json from the dashboard history.

    Each entry contains enough metadata for discovery plus the list of
    round task names so consumers can construct full key paths.
    """
    if _get_s3_client() is None:
        return False

    public_base_url = os.environ.get("R2_PUBLIC_URL", "")
    entries: list[dict[str, Any]] = []

    round_names_by_duel: dict[int, list[str]] = {}
    if latest_duel_dict:
        did = latest_duel_dict.get("duel_id")
        if did is not None:
            round_names_by_duel[did] = [
                r.get("task_name", "") for r in latest_duel_dict.get("rounds", [])
            ]

    for summary in duel_history:
        duel_id = summary.get("duel_id")
        if duel_id is None:
            continue
        round_task_names = round_names_by_duel.get(
            duel_id,
            [r.get("task_name", "") for r in summary.get("rounds", [])],
        )
        entries.append({
            "duel_id": duel_id,
            "started_at": summary.get("started_at"),
            "finished_at": summary.get("finished_at"),
            "king_repo": summary.get("king_repo"),
            "challenger_repo": summary.get("challenger_repo"),
            "king_replaced": summary.get("king_replaced", False),
            "wins": summary.get("wins", 0),
            "losses": summary.get("losses", 0),
            "ties": summary.get("ties", 0),
            "rounds": round_task_names,
            "path": f"{_DUELS_PREFIX}{duel_id:06d}/",
        })

    payload = {
        "updated_at": datetime.now(tz=UTC).isoformat(),
        "public_base_url": public_base_url,
        "duels": entries,
    }
    if _is_throttled():
        return False
    try:
        _upload_json(_INDEX_KEY, payload)
        log.info("Published duel index (%d entries) to R2", len(entries))
        return True
    except Exception as exc:
        if _is_throttle_error(exc):
            _note_throttle()
            return False
        log.exception("Failed to publish duel index to R2 (non-fatal)")
        return False


def backfill_duel_to_r2(
    duel_json_path: Path,
    tasks_root: Path,
    solution_labels: dict[str, str] | None = None,
) -> bool:
    """Upload a historical duel and its round artifacts to R2.

    Reads the full duel JSON from disk, uploads the duel record, then
    iterates over rounds and uploads each round's artifacts if available.
    Returns True if the duel record was uploaded.
    """
    if _get_s3_client() is None:
        log.warning("R2 credentials not configured; skipping backfill")
        return False

    duel_dict = json.loads(duel_json_path.read_text())
    duel_id = duel_dict["duel_id"]

    if not publish_duel_data(duel_id=duel_id, duel_dict=duel_dict):
        return False

    for round_data in duel_dict.get("rounds", []):
        task_name = round_data.get("task_name")
        if not task_name:
            continue
        try:
            publish_round_data(
                duel_id=duel_id, task_name=task_name,
                tasks_root=tasks_root, solution_labels=solution_labels,
            )
        except Exception:
            log.exception(
                "Backfill: failed to upload round %s for duel %d (non-fatal)",
                task_name, duel_id,
            )

    log.info("Backfilled duel %d from %s", duel_id, duel_json_path)
    return True


def publish_training_data(
    *,
    duel_id: int,
    duel_dict: dict[str, Any],
    tasks_root: Path,
    solution_labels: dict[str, str] | None = None,
) -> bool:
    """Emit a training.jsonl for the duel with one self-contained record per round.

    Each line contains the task prompt, reference diff, all solution diffs,
    comparison scores, solve metadata, and round outcome -- everything a
    training pipeline needs without chasing individual artifact files.

    Uploads to: sn66/duels/{duel_id}/training.jsonl
    """
    if _get_s3_client() is None:
        return False

    from workspace import build_solution_paths, build_task_paths

    labels = solution_labels or {}
    king_before = duel_dict.get("king_before", {})
    challenger_info = duel_dict.get("challenger", {})
    lines: list[str] = []

    for round_data in duel_dict.get("rounds", []):
        task_name = round_data.get("task_name")
        if not task_name or round_data.get("error"):
            continue

        task_paths = build_task_paths(tasks_root, task_name)
        record: dict[str, Any] = {
            "duel_id": duel_id,
            "task_name": task_name,
            "winner": round_data.get("winner"),
            "king_lines": round_data.get("king_lines", 0),
            "challenger_lines": round_data.get("challenger_lines", 0),
            "baseline_lines": round_data.get("baseline_lines", 0),
            "king_similarity_ratio": round_data.get("king_similarity_ratio", 0.0),
            "challenger_similarity_ratio": round_data.get("challenger_similarity_ratio", 0.0),
            "king_challenger_similarity": round_data.get("king_challenger_similarity", 0.0),
            "king_repo": king_before.get("repo_full_name"),
            "king_commit_sha": king_before.get("commit_sha"),
            "challenger_uid": challenger_info.get("uid"),
            "challenger_repo": challenger_info.get("repo_full_name"),
            "challenger_commit_sha": challenger_info.get("commit_sha"),
        }

        def _read_text(p: Path) -> str | None:
            try:
                return p.read_text() if p.exists() else None
            except Exception:
                return None

        def _read_json(p: Path) -> dict[str, Any] | None:
            try:
                return json.loads(p.read_text()) if p.exists() else None
            except Exception:
                return None

        task_json = _read_json(task_paths.task_json_path)
        if task_json:
            task_inner = task_json.get("task", {})
            record["repo"] = task_json.get("repo_full_name")
            record["commit_sha"] = task_json.get("commit_sha")
            record["task_prompt"] = task_inner.get("prompt_text")
            record["task_title"] = task_inner.get("title")

        record["reference_diff"] = _read_text(task_paths.reference_patch_path)

        for canonical in ("baseline", "king", "challenger"):
            disk_name = labels.get(canonical, canonical)
            sol_paths = build_solution_paths(task_paths, disk_name)
            record[f"{canonical}_diff"] = _read_text(sol_paths.solution_diff_path)

            solve_json = _read_json(sol_paths.solve_json_path)
            if solve_json:
                result = solve_json.get("result", {})
                record[f"{canonical}_elapsed_s"] = result.get("elapsed_seconds")
                record[f"{canonical}_model"] = result.get("model")
                record[f"{canonical}_exit_reason"] = result.get("exit_reason")
                record[f"{canonical}_tool_calls"] = result.get("tool_calls")
                usage = result.get("usage_summary") or {}
                record[f"{canonical}_total_tokens"] = usage.get("total_tokens") or result.get("total_tokens")
                record[f"{canonical}_cost"] = usage.get("cost") or result.get("cost")

        lines.append(json.dumps(record, separators=(",", ":")))

    if not lines:
        return False

    body = "\n".join(lines) + "\n"
    key = f"{_duel_key_prefix(duel_id)}training.jsonl"
    if _is_throttled():
        return False
    try:
        _upload_text(key, body, "application/x-ndjson")
        log.info("Published training data (%d rounds) for duel %d to R2", len(lines), duel_id)
        return True
    except Exception as exc:
        if _is_throttle_error(exc):
            _note_throttle()
            return False
        log.exception("Failed to publish training data for duel %d (non-fatal)", duel_id)
        return False


def fetch_chain_data(netuid: int) -> dict[str, Any] | None:
    """Fetch subnet and market data from the TaoMarketCap API."""
    api_key = os.environ.get("TMC_API_KEY")
    if not api_key:
        return None
    headers = {"Authorization": api_key, "Accept": "application/json"}
    base = "https://api.taomarketcap.com/public/v1"
    try:
        with httpx.Client(timeout=15, headers=headers) as c:
            market = c.get(f"{base}/market/market-data/")
            subnet = c.get(f"{base}/subnets/{netuid}/")
            weights = c.get(f"{base}/subnets/weights/{netuid}/")
        m = market.json() if market.status_code == 200 else {}
        s = subnet.json() if subnet.status_code == 200 else {}
        w = weights.json() if weights.status_code == 200 else {}
        snap = s.get("latest_snapshot", {})
        burn = int(snap.get("burn", 0))
        tao = float(m.get("current_price", 0))
        alpha_tao = float(snap.get("subnet_moving_price", 0))
        wt = []
        for we in w.get("weights", []):
            for tid, val in we.get("value", {}).items():
                wt.append({"validator_uid": we["uid"], "miner_uid": int(tid), "weight": val})
        return {
            "fetched_at": datetime.now(tz=UTC).isoformat(),
            "tao_price_usd": tao,
            "tao_change_24h": float((m.get("usd_quote") or {}).get("percent_change_24h", 0)),
            "tao_market_cap": float((m.get("usd_quote") or {}).get("market_cap", 0)),
            "alpha_price_tao": alpha_tao,
            "alpha_price_usd": alpha_tao * tao,
            "subnet_tao": int(snap.get("subnet_tao", 0)) / 1e9,
            "subnet_emission_per_day": int(snap.get("subnet_tao_in_emission", 0)) / 1e9 * 7200,
            "burn_cost_rao": burn,
            "burn_cost_tao": burn / 1e9,
            "burn_cost_usd": burn / 1e9 * tao,
            "neuron_count": int(snap.get("subnetwork_n", 0)),
            "max_neurons": int(snap.get("max_allowed_uids", 256)),
            "token_symbol": snap.get("token_symbol", ""),
            "subnet_name": (snap.get("subnet_identities_v3") or {}).get("subnetName", ""),
            "tempo": int(snap.get("tempo", 0)),
            "immunity_period": int(snap.get("immunity_period", 0)),
            "weights": wt,
        }
    except Exception:
        log.exception("Failed to fetch chain data (non-fatal)")
        return None
