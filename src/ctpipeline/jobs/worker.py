"""Polling worker process.

Run one worker per CPU slot, and for GPU nodes one worker per GPU with
``CTP_DEVICE=cuda:<n>`` (concurrency per GPU = 1 in this version). SIGTERM
finishes the current job and then exits (graceful shutdown).
"""

from __future__ import annotations

import os
import signal
import socket
import threading
import time
from types import FrameType

from ctpipeline.common.config import Settings, get_settings
from ctpipeline.common.logging import configure_logging, get_logger
from ctpipeline.jobs.db import init_db, make_engine, make_session_factory
from ctpipeline.jobs.runner import RunContext, run_job
from ctpipeline.jobs.store import JobStore
from ctpipeline.storage.objects import LocalObjectStore

log = get_logger(__name__)


class Worker:
    def __init__(self, ctx: RunContext, worker_id: str | None = None) -> None:
        self.ctx = ctx
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}"
        self.stop_event = threading.Event()
        self._last_recovery = 0.0

    def run_once(self) -> bool:
        """Process at most one job. Returns True if a job was processed."""
        now = time.monotonic()
        if now - self._last_recovery > min(60.0, self.ctx.settings.stale_job_seconds / 2):
            recovered = self.ctx.store.recover_stale(self.ctx.settings.stale_job_seconds)
            if recovered:
                log.warning("recovered stale jobs", extra={"count": len(recovered)})
            self._last_recovery = now
        job = self.ctx.store.claim_next(self.worker_id)
        if job is None:
            return False
        log.info("job claimed", extra={"job_id": job.id, "worker_id": self.worker_id})
        run_job(self.ctx, job.id)
        return True

    def run_forever(self) -> None:
        log.info("worker started", extra={"worker_id": self.worker_id})
        while not self.stop_event.is_set():
            try:
                busy = self.run_once()
            except Exception:
                log.exception("worker loop error")
                busy = False
            if not busy:
                self.stop_event.wait(self.ctx.settings.worker_poll_seconds)
        log.info("worker stopped", extra={"worker_id": self.worker_id})

    def stop(self, *_: object) -> None:
        self.stop_event.set()


def build_context(settings: Settings) -> RunContext:
    if settings.database_url.startswith("sqlite:///"):
        db_path = settings.database_url.removeprefix("sqlite:///")
        if db_path and db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    engine = make_engine(settings.database_url)
    init_db(engine)
    return RunContext(
        store=JobStore(make_session_factory(engine)),
        objects=LocalObjectStore(settings.storage_root),
        settings=settings,
    )


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    worker = Worker(build_context(settings))

    def _handle(signum: int, _frame: FrameType | None) -> None:
        log.info("shutdown requested", extra={"signal": signum})
        worker.stop()

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)
    worker.run_forever()


if __name__ == "__main__":
    main()
