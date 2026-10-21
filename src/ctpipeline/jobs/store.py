"""Job persistence: lifecycle transitions, idempotency, claiming, retries.

The database is the source of truth for the queue: a worker *claims* a job
with a conditional UPDATE (``status = 'QUEUED'``), so two workers can never
run the same job, and jobs survive restarts of any component.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, cast

from sqlalchemy import CursorResult, and_, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from ctpipeline.common.errors import InvalidTransitionError, PipelineError, TransientError
from ctpipeline.jobs.db import Artifact, AuditEvent, Job, JobEvent, utcnow
from ctpipeline.jobs.states import RUNNING, TERMINAL, JobStatus, check_transition


class JobNotFoundError(PipelineError):
    code = "JOB_NOT_FOUND"


class JobStore:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._sf = session_factory

    def session(self) -> Session:
        return self._sf()

    # -- creation ---------------------------------------------------------
    def create_job(
        self,
        *,
        model_name: str,
        input_prefix: str,
        idempotency_key: str | None = None,
        priority: int = 0,
        requested_series_uid: str | None = None,
        params: dict[str, Any] | None = None,
        max_attempts: int = 3,
        job_id: str | None = None,
    ) -> tuple[Job, bool]:
        """Create a job in RECEIVED state. Returns ``(job, created)``.

        With an idempotency key, a repeated request returns the existing job.
        """
        if idempotency_key:
            existing = self.find_by_idempotency_key(idempotency_key)
            if existing is not None:
                return existing, False
        job = Job(
            status=JobStatus.RECEIVED.value,
            model_name=model_name,
            input_prefix=input_prefix,
            idempotency_key=idempotency_key,
            priority=priority,
            requested_series_uid=requested_series_uid,
            params=params or {},
            max_attempts=max_attempts,
        )
        if job_id:
            job.id = job_id
        with self.session() as s:
            s.add(job)
            s.add(JobEvent(job_id=job.id, from_status=None, to_status=job.status))
            try:
                s.commit()
            except IntegrityError:
                s.rollback()
                if idempotency_key:
                    existing = self.find_by_idempotency_key(idempotency_key)
                    if existing is not None:
                        return existing, False
                raise
        return job, True

    def find_by_idempotency_key(self, key: str) -> Job | None:
        with self.session() as s:
            return s.scalar(select(Job).where(Job.idempotency_key == key))

    def get(self, job_id: str) -> Job:
        with self.session() as s:
            job = s.get(Job, job_id)
            if job is None:
                raise JobNotFoundError("Job not found")
            return job

    def set_fields(self, job_id: str, **fields: Any) -> Job:
        with self.session() as s:
            job = s.get(Job, job_id)
            if job is None:
                raise JobNotFoundError("Job not found")
            for key, value in fields.items():
                setattr(job, key, value)
            s.commit()
            return job

    # -- transitions -------------------------------------------------------
    def transition(
        self,
        job_id: str,
        target: JobStatus,
        *,
        detail: dict[str, Any] | None = None,
        **fields: Any,
    ) -> Job:
        with self.session() as s:
            job = s.get(Job, job_id, with_for_update=True)
            if job is None:
                raise JobNotFoundError("Job not found")
            check_transition(job.status, target)
            now = utcnow()
            previous = job.status
            job.status = target.value
            job.heartbeat_at = now
            if target == JobStatus.QUEUED:
                job.queued_at = now
            if target in TERMINAL:
                job.finished_at = now
            for key, value in fields.items():
                setattr(job, key, value)
            s.add(
                JobEvent(
                    job_id=job_id, from_status=previous, to_status=target.value, detail=detail or {}
                )
            )
            s.commit()
            return job

    def heartbeat(self, job_id: str) -> None:
        with self.session() as s:
            s.execute(update(Job).where(Job.id == job_id).values(heartbeat_at=utcnow()))
            s.commit()

    # -- queue -------------------------------------------------------------
    def claim_next(self, worker_id: str, now: datetime | None = None) -> Job | None:
        """Atomically move the highest-priority eligible QUEUED job to PREPROCESSING."""
        now = now or utcnow()
        with self.session() as s:
            candidates = s.scalars(
                select(Job.id)
                .where(
                    Job.status == JobStatus.QUEUED.value,
                    Job.cancel_requested.is_(False),
                    or_(Job.not_before.is_(None), Job.not_before <= now),
                )
                .order_by(Job.priority.desc(), Job.created_at.asc())
                .limit(10)
            ).all()
            for job_id in candidates:
                result = cast(
                    CursorResult[Any],
                    s.execute(
                        update(Job)
                        .where(and_(Job.id == job_id, Job.status == JobStatus.QUEUED.value))
                        .values(
                            status=JobStatus.PREPROCESSING.value,
                            attempts=Job.attempts + 1,
                            worker_id=worker_id,
                            started_at=now,
                            heartbeat_at=now,
                        )
                    ),
                )
                if result.rowcount == 1:
                    s.add(
                        JobEvent(
                            job_id=job_id,
                            from_status=JobStatus.QUEUED.value,
                            to_status=JobStatus.PREPROCESSING.value,
                            detail={"worker_id": worker_id},
                        )
                    )
                    s.commit()
                    return self.get(job_id)
                s.rollback()
        return None

    def record_failure(self, job_id: str, error: PipelineError, backoff_seconds: float) -> Job:
        """Retry retryable errors with exponential backoff; otherwise fail.

        A retryable error that exhausted ``max_attempts`` is dead-lettered.
        """
        job = self.get(job_id)
        if job.status in {s.value for s in TERMINAL}:
            return job
        if error.retryable and job.attempts < job.max_attempts:
            delay = backoff_seconds * (2 ** max(job.attempts - 1, 0))
            return self.transition(
                job_id,
                JobStatus.QUEUED,
                detail={"retry": True, **error.to_dict()},
                not_before=utcnow() + timedelta(seconds=delay),
                error_code=error.code,
                error_message=error.message,
            )
        return self.transition(
            job_id,
            JobStatus.FAILED,
            detail=error.to_dict(),
            error_code=error.code,
            error_message=error.message,
            dead_lettered=error.retryable,
        )

    def request_cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        status = JobStatus(job.status)
        if status in TERMINAL:
            raise InvalidTransitionError(f"Job is already {status}")
        if status in RUNNING:
            return self.set_fields(job_id, cancel_requested=True)
        return self.transition(
            job_id, JobStatus.CANCELLED, detail={"reason": "user"}, cancel_requested=True
        )

    def is_cancel_requested(self, job_id: str) -> bool:
        return bool(self.get(job_id).cancel_requested)

    def recover_stale(self, stale_after_seconds: float, now: datetime | None = None) -> list[str]:
        """Requeue running jobs whose worker stopped heart-beating (crash/restart)."""
        now = now or utcnow()
        cutoff = now - timedelta(seconds=stale_after_seconds)
        with self.session() as s:
            stale = s.scalars(
                select(Job.id).where(
                    Job.status.in_([st.value for st in RUNNING]),
                    or_(Job.heartbeat_at.is_(None), Job.heartbeat_at < cutoff),
                )
            ).all()
        recovered = []
        for job_id in stale:
            try:
                job = self.get(job_id)
                if job.cancel_requested:
                    self.transition(job_id, JobStatus.CANCELLED, detail={"reason": "recovery"})
                else:
                    self.record_failure(
                        job_id,
                        TransientError("Worker lost during processing", code="WORKER_LOST"),
                        backoff_seconds=0,
                    )
                recovered.append(job_id)
            except InvalidTransitionError:
                continue
        return recovered

    # -- artifacts & audit --------------------------------------------------
    def replace_artifacts(self, job_id: str, artifacts: list[dict[str, Any]]) -> None:
        with self.session() as s:
            for old in s.scalars(select(Artifact).where(Artifact.job_id == job_id)).all():
                s.delete(old)
            for art in artifacts:
                s.add(Artifact(job_id=job_id, **art))
            s.commit()

    def artifacts(self, job_id: str) -> list[Artifact]:
        with self.session() as s:
            return list(s.scalars(select(Artifact).where(Artifact.job_id == job_id)).all())

    def events(self, job_id: str) -> list[JobEvent]:
        with self.session() as s:
            return list(
                s.scalars(
                    select(JobEvent).where(JobEvent.job_id == job_id).order_by(JobEvent.id)
                ).all()
            )

    def audit(
        self,
        action: str,
        actor: str,
        job_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        with self.session() as s:
            s.add(AuditEvent(action=action, actor=actor, job_id=job_id, detail=detail or {}))
            s.commit()

    def queue_depth(self) -> int:
        with self.session() as s:
            stmt = select(func.count()).where(Job.status == JobStatus.QUEUED.value)
            return int(s.scalar(stmt) or 0)
