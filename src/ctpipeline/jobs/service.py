"""Job submission: store inputs, validate, enqueue."""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from typing import Any

from ctpipeline.common.errors import IngestionError
from ctpipeline.ingestion.loader import IngestionOptions, load_series
from ctpipeline.jobs.db import Job
from ctpipeline.jobs.profile import load_profile
from ctpipeline.jobs.runner import RunContext
from ctpipeline.jobs.states import JobStatus


def submit_study(
    ctx: RunContext,
    files: Iterable[bytes],
    *,
    model_name: str,
    idempotency_key: str | None = None,
    series_instance_uid: str | None = None,
    priority: int = 0,
    model_params: dict[str, Any] | None = None,
) -> tuple[Job, bool]:
    """Create a job for the uploaded instances. Returns ``(job, created)``.

    Invalid input ends in FAILED with a structured error code rather than
    raising, so the client can always look the job up.
    """
    if idempotency_key:
        existing = ctx.store.find_by_idempotency_key(idempotency_key)
        if existing is not None:
            return existing, False
    load_profile(ctx.settings.configs_dir, model_name)  # unknown model -> ModelError

    job_id = str(uuid.uuid4())
    prefix = f"jobs/{job_id}/input"
    job, created = ctx.store.create_job(
        job_id=job_id,
        model_name=model_name,
        input_prefix=prefix,
        idempotency_key=idempotency_key,
        priority=priority,
        requested_series_uid=series_instance_uid,
        params={"model_params": model_params or {}},
        max_attempts=ctx.settings.max_attempts,
    )
    if not created:
        return job, False

    count = 0
    for index, data in enumerate(files):
        ctx.objects.put(f"{prefix}/{index:06d}.dcm", data)
        count += 1
    ctx.store.transition(job_id, JobStatus.VALIDATING, input_file_count=count)
    try:
        paths = [ctx.objects.local_path(f"{prefix}/{i:06d}.dcm") for i in range(count)]
        series = load_series(
            paths,
            IngestionOptions(series_instance_uid=series_instance_uid, load_pixels=False),
        )
    except IngestionError as exc:
        job = ctx.store.transition(
            job_id,
            JobStatus.FAILED,
            detail=exc.to_dict(),
            error_code=exc.code,
            error_message=exc.message,
        )
        return job, True
    job = ctx.store.transition(
        job_id,
        JobStatus.QUEUED,
        study_instance_uid=series.study_instance_uid,
        series_instance_uid=series.series_instance_uid,
        input_manifest_sha256=series.input_manifest_sha256,
        warnings=series.warnings,
    )
    return job, True
