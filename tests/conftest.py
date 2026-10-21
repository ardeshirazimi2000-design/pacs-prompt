from __future__ import annotations

from pathlib import Path

import pytest

from ctpipeline.common.config import Settings
from ctpipeline.jobs.runner import RunContext
from ctpipeline.jobs.worker import build_context
from ctpipeline.testing.synthetic import make_ct_series

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url=f"sqlite:///{tmp_path / 'db.sqlite'}",
        storage_root=tmp_path / "objects",
        configs_dir=REPO_ROOT / "configs",
        retry_backoff_seconds=0,
        max_attempts=3,
    )


@pytest.fixture
def ctx(settings: Settings) -> RunContext:
    return build_context(settings)


@pytest.fixture
def series_paths(tmp_path: Path) -> list[Path]:
    return make_ct_series(tmp_path / "series")


@pytest.fixture
def series_bytes(series_paths: list[Path]) -> list[bytes]:
    return [p.read_bytes() for p in series_paths]
