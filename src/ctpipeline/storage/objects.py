"""Object storage abstraction.

``LocalObjectStore`` keeps objects on a filesystem path (a PVC in Kubernetes).
An S3-compatible implementation can be added behind the same interface.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Protocol

from ctpipeline.common.hashing import sha256_bytes


class ObjectStore(Protocol):
    def put(self, key: str, data: bytes) -> str: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def list(self, prefix: str) -> list[str]: ...
    def local_path(self, key: str) -> Path: ...


class LocalObjectStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root not in path.parents:
            raise ValueError("Object key escapes the storage root")
        return path

    def put(self, key: str, data: bytes) -> str:
        """Atomically write ``data``; returns its SHA-256."""
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return sha256_bytes(data)

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list(self, prefix: str) -> list[str]:
        base = self._path(prefix)
        if not base.is_dir():
            return []
        return sorted(
            str(p.relative_to(self.root))
            for p in base.rglob("*")
            if p.is_file() and not p.name.startswith(".tmp-")
        )

    def local_path(self, key: str) -> Path:
        return self._path(key)
