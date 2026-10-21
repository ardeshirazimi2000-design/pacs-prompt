"""HTTP API. Contains no image-processing or model logic; it only stores
uploads, validates them, and reads job state written by workers."""

from __future__ import annotations

import secrets
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel
from sqlalchemy import func, select, text

from ctpipeline import __version__
from ctpipeline.common.config import Settings, get_settings
from ctpipeline.common.errors import InvalidTransitionError, ModelError
from ctpipeline.common.logging import configure_logging, get_logger
from ctpipeline.jobs.db import Job
from ctpipeline.jobs.runner import RunContext
from ctpipeline.jobs.service import submit_study
from ctpipeline.jobs.states import JobStatus
from ctpipeline.jobs.store import JobNotFoundError
from ctpipeline.jobs.worker import Worker, build_context

log = get_logger(__name__)

DISCLAIMER = "Research prototype. Outputs are not for diagnostic use."


class ErrorInfo(BaseModel):
    code: str
    message: str


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    model: str
    priority: int
    attempts: int
    max_attempts: int
    cancel_requested: bool
    dead_lettered: bool
    series_instance_uid: str | None
    input_file_count: int
    warnings: list[str]
    error: ErrorInfo | None
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None


class ArtifactResponse(BaseModel):
    type: str
    sha256: str
    size: int
    content_type: str
    url: str


class ResultsResponse(BaseModel):
    job_id: str
    status: JobStatus
    disclaimer: str
    artifacts: list[ArtifactResponse]
    manifest: dict[str, Any] | None


class EventResponse(BaseModel):
    from_status: str | None
    to_status: str
    at: datetime
    detail: dict[str, Any]


def to_response(job: Job) -> JobResponse:
    return JobResponse(
        job_id=job.id,
        status=JobStatus(job.status),
        model=job.model_name,
        priority=job.priority,
        attempts=job.attempts,
        max_attempts=job.max_attempts,
        cancel_requested=job.cancel_requested,
        dead_lettered=job.dead_lettered,
        series_instance_uid=job.series_instance_uid,
        input_file_count=job.input_file_count,
        warnings=list(job.warnings or []),
        error=ErrorInfo(code=job.error_code, message=job.error_message or "")
        if job.error_code
        else None,
        created_at=job.created_at,
        updated_at=job.updated_at,
        finished_at=job.finished_at,
    )


def get_ctx(request: Request) -> RunContext:
    ctx: RunContext = request.app.state.ctx
    return ctx


def get_actor(request: Request, x_api_key: Annotated[str | None, Header()] = None) -> str:
    expected = request.app.state.ctx.settings.api_key
    if expected is None:
        return "anonymous"
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return "api-key"


Ctx = Annotated[RunContext, Depends(get_ctx)]
Actor = Annotated[str, Depends(get_actor)]


def create_app(ctx: RunContext | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or (ctx.settings if ctx else get_settings())
    context = ctx or build_context(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        worker: Worker | None = None
        thread: threading.Thread | None = None
        if settings.embedded_worker:
            worker = Worker(context, worker_id="embedded")
            thread = threading.Thread(target=worker.run_forever, daemon=True)
            thread.start()
        yield
        if worker and thread:
            worker.stop()
            thread.join(timeout=30)

    app = FastAPI(
        title="CT AI Pipeline",
        version=__version__,
        description=DISCLAIMER,
        lifespan=lifespan,
    )
    app.state.ctx = context

    def load_job(c: RunContext, job_id: str) -> Job:
        try:
            return c.store.get(job_id)
        except JobNotFoundError:
            raise HTTPException(status_code=404, detail="Job not found") from None

    @app.post("/v1/studies", response_model=JobResponse, status_code=202)
    async def create_study_job(
        c: Ctx,
        who: Actor,
        response: Response,
        files: Annotated[list[UploadFile], File(description="DICOM instances of one CT series")],
        model: Annotated[str | None, Form()] = None,
        series_instance_uid: Annotated[str | None, Form()] = None,
        priority: Annotated[int, Form(ge=-10, le=10)] = 0,
        idempotency_key: Annotated[str | None, Header(max_length=128)] = None,
    ) -> JobResponse:
        if not files:
            raise HTTPException(status_code=422, detail="No files uploaded")
        if len(files) > settings.max_files_per_study:
            raise HTTPException(status_code=413, detail="Too many files")
        payloads: list[bytes] = []
        total = 0
        for upload in files:
            data = await upload.read()
            total += len(data)
            if total > settings.max_upload_bytes:
                raise HTTPException(status_code=413, detail="Upload too large")
            payloads.append(data)
        try:
            job, created = await run_in_threadpool(
                submit_study,
                c,
                payloads,
                model_name=model or settings.default_model,
                idempotency_key=idempotency_key,
                series_instance_uid=series_instance_uid,
                priority=priority,
            )
        except ModelError as exc:
            raise HTTPException(status_code=422, detail=exc.to_dict()) from None
        if not created:
            response.status_code = 200
        c.store.audit("create_job", who, job.id, {"created": created})
        return to_response(job)

    @app.get("/v1/jobs/{job_id}", response_model=JobResponse)
    def get_job(job_id: str, c: Ctx, _who: Actor) -> JobResponse:
        return to_response(load_job(c, job_id))

    @app.get("/v1/jobs/{job_id}/events", response_model=list[EventResponse])
    def get_job_events(job_id: str, c: Ctx, _who: Actor) -> list[EventResponse]:
        load_job(c, job_id)
        return [
            EventResponse(
                from_status=e.from_status, to_status=e.to_status, at=e.at, detail=e.detail or {}
            )
            for e in c.store.events(job_id)
        ]

    @app.get("/v1/jobs/{job_id}/results", response_model=ResultsResponse)
    def get_results(job_id: str, c: Ctx, _who: Actor, request: Request) -> ResultsResponse:
        job = load_job(c, job_id)
        if job.status != JobStatus.SUCCEEDED.value:
            raise HTTPException(status_code=409, detail=f"Job is {job.status}")
        artifacts = [
            ArtifactResponse(
                type=a.type,
                sha256=a.sha256,
                size=a.size,
                content_type=a.content_type,
                url=str(request.url_for("download_result", job_id=job_id, artifact_type=a.type)),
            )
            for a in c.store.artifacts(job_id)
        ]
        return ResultsResponse(
            job_id=job_id,
            status=JobStatus(job.status),
            disclaimer=DISCLAIMER,
            artifacts=artifacts,
            manifest=job.manifest,
        )

    @app.get("/v1/jobs/{job_id}/results/{artifact_type}", name="download_result")
    def download_result(job_id: str, artifact_type: str, c: Ctx, who: Actor) -> Response:
        job = load_job(c, job_id)
        if job.status != JobStatus.SUCCEEDED.value:
            raise HTTPException(status_code=409, detail=f"Job is {job.status}")
        match = [a for a in c.store.artifacts(job_id) if a.type == artifact_type.upper()]
        if not match:
            raise HTTPException(status_code=404, detail="Artifact not found")
        art = match[0]
        c.store.audit("download_result", who, job_id, {"type": art.type})
        filename = art.key.rsplit("/", 1)[-1]
        return Response(
            content=c.objects.get(art.key),
            media_type=art.content_type,
            headers={
                "Content-Disposition": f'attachment; filename="{job_id}-{filename}"',
                "X-Content-SHA256": art.sha256,
            },
        )

    @app.post("/v1/jobs/{job_id}/cancel", response_model=JobResponse)
    def cancel_job(job_id: str, c: Ctx, who: Actor) -> JobResponse:
        load_job(c, job_id)
        try:
            job = c.store.request_cancel(job_id)
        except InvalidTransitionError as exc:
            raise HTTPException(status_code=409, detail=exc.message) from None
        c.store.audit("cancel_job", who, job_id)
        return to_response(job)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz(c: Ctx) -> dict[str, str]:
        try:
            with c.store.session() as s:
                s.execute(text("SELECT 1"))
            c.objects.put(".readyz", b"ok")
        except Exception:
            log.exception("readiness check failed")
            raise HTTPException(status_code=503, detail="not ready") from None
        return {"status": "ready"}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics(c: Ctx) -> str:
        with c.store.session() as s:
            rows = s.execute(select(Job.status, func.count()).group_by(Job.status)).all()
        lines = [
            "# HELP ctpipeline_jobs Number of jobs by status.",
            "# TYPE ctpipeline_jobs gauge",
        ]
        counts = {status: 0 for status in JobStatus}
        counts.update({JobStatus(status): n for status, n in rows})
        lines += [f'ctpipeline_jobs{{status="{k.value}"}} {v}' for k, v in counts.items()]
        lines += [
            "# HELP ctpipeline_queue_depth Jobs waiting in the queue.",
            "# TYPE ctpipeline_queue_depth gauge",
            f"ctpipeline_queue_depth {counts[JobStatus.QUEUED]}",
        ]
        return "\n".join(lines) + "\n"

    return app


def build_app() -> FastAPI:
    """Uvicorn factory: ``uvicorn ctpipeline.api.app:build_app --factory``."""
    settings = get_settings()
    configure_logging(settings.log_level)
    return create_app(settings=settings)
