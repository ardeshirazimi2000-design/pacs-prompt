from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings, read from environment variables prefixed ``CTP_``."""

    model_config = SettingsConfigDict(env_prefix="CTP_", env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./data/ctpipeline.db"
    storage_root: Path = Path("./data/objects")
    configs_dir: Path = Path("./configs")
    log_level: str = "INFO"

    # API
    api_key: str | None = None
    max_upload_bytes: int = 2 * 1024**3
    max_files_per_study: int = 5000

    # Jobs / queue
    max_attempts: int = 3
    retry_backoff_seconds: float = 5.0
    stale_job_seconds: float = 900.0
    worker_poll_seconds: float = 1.0
    embedded_worker: bool = False

    # Inference
    default_model: str = "dummy-threshold"
    device: str = "cpu"

    # Provenance
    code_revision: str = "unknown"
    container_digest: str = "unknown"

    @field_validator("api_key", mode="before")
    @classmethod
    def _empty_api_key_disables_auth(cls, value: object) -> object:
        return None if value == "" else value


@lru_cache
def get_settings() -> Settings:
    return Settings()
