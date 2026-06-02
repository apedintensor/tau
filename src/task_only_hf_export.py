from __future__ import annotations

import fnmatch
import gzip
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SOURCE_PATTERNS = ("tasks/**/*.jsonl", "tasks/**/*.jsonl.gz", "*.jsonl", "*.jsonl.gz")
DEFAULT_TARGET_PATH = "tasks"
DEFAULT_SHARD_MAX_BYTES = 500_000_000
TASK_ARTIFACT_PREFIX = "task/"


@dataclass(frozen=True, slots=True)
class TaskOnlyExportResult:
    target_dataset: str
    path_in_repo: str
    source_datasets: tuple[str, ...]
    source_files: int
    uploaded_files: int
    scanned_rows: int
    exported_rows: int
    skipped_rows: int
    upload_url: str | None


def task_only_artifacts(row: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = row.get("artifacts")
    if not isinstance(artifacts, list):
        return []
    return [
        dict(artifact)
        for artifact in artifacts
        if isinstance(artifact, dict) and str(artifact.get("path") or "").startswith(TASK_ARTIFACT_PREFIX)
    ]


def artifact_bundle_sha256(artifacts: Sequence[dict[str, Any]]) -> str:
    payload = json.dumps(list(artifacts), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def task_only_row(
    row: dict[str, Any],
    *,
    source_dataset: str,
    source_path: str,
) -> dict[str, Any] | None:
    task_name = row.get("task_name")
    if not isinstance(task_name, str) or not task_name:
        return None
    artifacts = task_only_artifacts(row)
    if not artifacts:
        return None
    return {
        "schema_version": 1,
        "row_type": "tau_task_only",
        "source": {
            "dataset": source_dataset,
            "path": source_path,
            "archive_hour": row.get("archive_hour"),
            "archive_reason": row.get("archive_reason"),
        },
        "task_name": task_name,
        "task_root_name": row.get("task_root_name"),
        "task_metadata": row.get("task_metadata"),
        "commit_metadata": row.get("commit_metadata"),
        "artifact_count": len(artifacts),
        "artifact_bundle_sha256": artifact_bundle_sha256(artifacts),
        "artifacts": artifacts,
    }


def repo_files_matching(files: Iterable[str], patterns: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(path for path in files if any(fnmatch.fnmatch(path, pattern) for pattern in patterns)))


def iter_jsonl_path(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                yield payload


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def shard_path(path_in_repo: str, shard_index: int) -> str:
    normalized = path_in_repo.strip("/")
    if normalized.endswith(".jsonl") or normalized.endswith(".jsonl.gz"):
        stem = normalized.removesuffix(".gz").removesuffix(".jsonl")
        return f"{stem}-{shard_index:05d}.jsonl.gz"
    return f"{normalized}/tasks-{shard_index:05d}.jsonl.gz"


def upload_task_only_shards(
    *,
    rows: Iterable[dict[str, Any]],
    target_dataset: str,
    token: str,
    path_in_repo: str,
    max_uncompressed_bytes: int,
    dry_run: bool,
) -> tuple[int, int, str | None]:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    exported = 0
    uploaded_files = 0
    upload_url = None
    with tempfile.TemporaryDirectory() as td:
        tmp_root = Path(td)
        shard_index = 0
        handle: gzip.GzipFile | None = None
        current_path: Path | None = None
        current_bytes = 0

        def open_next_shard() -> gzip.GzipFile:
            nonlocal current_path, current_bytes, shard_index
            current_path = tmp_root / shard_path(path_in_repo, shard_index)
            current_path.parent.mkdir(parents=True, exist_ok=True)
            current_bytes = 0
            shard_index += 1
            return gzip.open(current_path, "wt", encoding="utf-8")

        def close_shard() -> None:
            nonlocal handle, current_path, uploaded_files
            if handle is None or current_path is None:
                return
            handle.close()
            handle = None
            if current_bytes <= 0:
                current_path.unlink(missing_ok=True)
                return
            uploaded_files += 1

        for row in rows:
            encoded = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            encoded_bytes = len(encoded.encode("utf-8"))
            if handle is None:
                handle = open_next_shard()
            if current_bytes > 0 and current_bytes + encoded_bytes > max_uncompressed_bytes:
                close_shard()
                handle = open_next_shard()
            handle.write(encoded)
            current_bytes += encoded_bytes
            exported += 1

        close_shard()

        if uploaded_files > 0 and not dry_run:
            upload = api.upload_folder(
                folder_path=str(tmp_root),
                path_in_repo="",
                repo_id=target_dataset,
                repo_type="dataset",
                commit_message=f"Publish {exported} tau task-only rows in {uploaded_files} shard(s)",
            )
            upload_url = getattr(upload, "commit_url", None) or str(upload or "")

    return exported, uploaded_files, upload_url


def dedupe_task_rows(rows: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    seen: set[str] = set()
    for row in rows:
        task_name = row.get("task_name")
        if not isinstance(task_name, str) or task_name in seen:
            continue
        seen.add(task_name)
        yield row


def iter_hf_task_only_rows(
    *,
    source_datasets: Sequence[str],
    token: str,
    patterns: Sequence[str] = DEFAULT_SOURCE_PATTERNS,
    limit: int | None = None,
) -> Iterator[tuple[dict[str, Any], str, str]]:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    yielded = 0
    for dataset_id in source_datasets:
        files = repo_files_matching(api.list_repo_files(dataset_id, repo_type="dataset"), patterns)
        for path_in_repo in files:
            local_path = Path(
                hf_hub_download(
                    repo_id=dataset_id,
                    filename=path_in_repo,
                    repo_type="dataset",
                    token=token,
                )
            )
            for row in iter_jsonl_path(local_path):
                task_row = task_only_row(row, source_dataset=dataset_id, source_path=path_in_repo)
                if task_row is None:
                    continue
                yield task_row, dataset_id, path_in_repo
                yielded += 1
                if limit is not None and yielded >= limit:
                    return


def create_task_only_hf_repo(
    *,
    source_datasets: Sequence[str],
    target_dataset: str,
    token: str | None = None,
    token_env: str = "HF_TOKEN",
    source_patterns: Sequence[str] = DEFAULT_SOURCE_PATTERNS,
    path_in_repo: str = DEFAULT_TARGET_PATH,
    shard_max_bytes: int = DEFAULT_SHARD_MAX_BYTES,
    private: bool = True,
    limit: int | None = None,
    dry_run: bool = False,
) -> TaskOnlyExportResult:
    from huggingface_hub import HfApi

    resolved_token = token or os.environ.get(token_env)
    if not resolved_token:
        raise RuntimeError(f"${token_env} is required to read source datasets and write the target dataset")
    if not source_datasets:
        raise ValueError("at least one --source-dataset is required")

    scanned = 0
    source_file_keys: set[tuple[str, str]] = set()

    def projected_rows() -> Iterator[dict[str, Any]]:
        nonlocal scanned
        for row, dataset_id, source_path in iter_hf_task_only_rows(
            source_datasets=source_datasets,
            token=resolved_token,
            patterns=source_patterns,
            limit=limit,
        ):
            scanned += 1
            source_file_keys.add((dataset_id, source_path))
            yield row

    if not dry_run:
        api = HfApi(token=resolved_token)
        api.create_repo(
            repo_id=target_dataset,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
    exported, uploaded_files, upload_url = upload_task_only_shards(
        rows=dedupe_task_rows(projected_rows()),
        target_dataset=target_dataset,
        token=resolved_token,
        path_in_repo=path_in_repo,
        max_uncompressed_bytes=max(1, shard_max_bytes),
        dry_run=dry_run,
    )

    return TaskOnlyExportResult(
        target_dataset=target_dataset,
        path_in_repo=path_in_repo,
        source_datasets=tuple(source_datasets),
        source_files=len(source_file_keys),
        uploaded_files=uploaded_files,
        scanned_rows=scanned,
        exported_rows=exported,
        skipped_rows=max(0, scanned - exported),
        upload_url=upload_url,
    )
