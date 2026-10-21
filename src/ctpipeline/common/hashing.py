from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_json(obj: Any) -> str:
    """Stable hash of a JSON-serialisable object (key order independent)."""
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def sha256_manifest(items: Iterable[tuple[str, str]]) -> str:
    """Hash of a set of (identifier, content-hash) pairs, order independent."""
    lines = "\n".join(f"{key}:{value}" for key, value in sorted(items))
    return sha256_bytes(lines.encode("utf-8"))
