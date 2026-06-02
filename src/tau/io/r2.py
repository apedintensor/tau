from __future__ import annotations

import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class S3Client(ABC):
    @abstractmethod
    def put_object(self, *, Bucket: str, Key: str, Body: bytes | str, **kwargs: Any) -> dict: ...
    @abstractmethod
    def get_object(self, *, Bucket: str, Key: str, **kwargs: Any) -> dict: ...
    @abstractmethod
    def delete_object(self, *, Bucket: str, Key: str, **kwargs: Any) -> dict: ...
    @abstractmethod
    def delete_objects(self, *, Bucket: str, Delete: dict, **kwargs: Any) -> dict: ...
    @abstractmethod
    def list_objects_v2(self, *, Bucket: str, **kwargs: Any) -> dict: ...


class BotoS3Client(S3Client):
    """Thin wrapper around a boto3 S3 client."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def put_object(self, *, Bucket: str, Key: str, Body: bytes | str, **kwargs: Any) -> dict:
        return self._client.put_object(Bucket=Bucket, Key=Key, Body=Body, **kwargs)

    def get_object(self, *, Bucket: str, Key: str, **kwargs: Any) -> dict:
        return self._client.get_object(Bucket=Bucket, Key=Key, **kwargs)

    def delete_object(self, *, Bucket: str, Key: str, **kwargs: Any) -> dict:
        return self._client.delete_object(Bucket=Bucket, Key=Key, **kwargs)

    def delete_objects(self, *, Bucket: str, Delete: dict, **kwargs: Any) -> dict:
        return self._client.delete_objects(Bucket=Bucket, Delete=Delete, **kwargs)

    def list_objects_v2(self, *, Bucket: str, **kwargs: Any) -> dict:
        return self._client.list_objects_v2(Bucket=Bucket, **kwargs)


class LocalS3Client(S3Client):
    """Writes to a local directory instead of uploading to R2. Useful for tests and replay bench."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def _path(self, bucket: str, key: str) -> Path:
        return self._root / bucket / key

    def put_object(self, *, Bucket: str, Key: str, Body: bytes | str, **kwargs: Any) -> dict:
        path = self._path(Bucket, Key)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(Body, str):
            path.write_text(Body, encoding="utf-8")
        else:
            path.write_bytes(Body)
        return {}

    def get_object(self, *, Bucket: str, Key: str, **kwargs: Any) -> dict:
        data = self._path(Bucket, Key).read_bytes()
        return {"Body": io.BytesIO(data)}

    def delete_object(self, *, Bucket: str, Key: str, **kwargs: Any) -> dict:
        self._path(Bucket, Key).unlink(missing_ok=True)
        return {}

    def delete_objects(self, *, Bucket: str, Delete: dict, **kwargs: Any) -> dict:
        for obj in Delete.get("Objects", []):
            self.delete_object(Bucket=Bucket, Key=obj["Key"])
        return {"Deleted": Delete.get("Objects", []), "Errors": []}

    def list_objects_v2(self, *, Bucket: str, **kwargs: Any) -> dict:
        prefix = kwargs.get("Prefix", "")
        bucket_root = self._root / Bucket
        if not bucket_root.exists():
            return {"Contents": [], "IsTruncated": False}
        contents = [
            {"Key": str(p.relative_to(bucket_root)), "Size": p.stat().st_size}
            for p in bucket_root.rglob("*")
            if p.is_file() and str(p.relative_to(bucket_root)).startswith(prefix)
        ]
        return {"Contents": contents, "IsTruncated": False}
