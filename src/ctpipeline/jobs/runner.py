"""Executes one claimed job through all pipeline stages.

Stage outputs are written under deterministic keys (``jobs/<id>/outputs/``),
so a retried job overwrites partial results instead of duplicating them, and
the job is only marked SUCCEEDED after every artifact and the manifest exist.
"""

from __future__ import annotations

import io
import json
import platform
import time
from dataclasses import dataclass
from typing import Any

import highdicom
import numpy as np
import pydicom
import scipy

from ctpipeline import __version__
from ctpipeline.adapters.registry import create_adapter
from ctpipeline.common.config import Settings
from ctpipeline.common.errors import PipelineError, TransientError
from ctpipeline.common.hashing import sha256_bytes
from ctpipeline.common.logging import get_logger
from ctpipeline.ingestion.loader import IngestionOptions, load_series
from ctpipeline.jobs.profile import PipelineProfile, load_profile
from ctpipeline.jobs.states import JobStatus
from ctpipeline.jobs.store import JobStore
from ctpipeline.outputs.common import ExportContext
from ctpipeline.outputs.dicom_seg import build_segmentation
from ctpipeline.outputs.dicom_sr import build_measurement_report
from ctpipeline.outputs.overlay import render_overlay
from ctpipeline.postprocessing.pipeline import postprocess
from ctpipeline.preprocessing.pipeline import preprocess
from ctpipeline.storage.objects import ObjectStore

log = get_logger(__name__)


class JobCancelledError(Exception):
    pass


@dataclass
class RunContext:
    store: JobStore
    objects: ObjectStore
    settings: Settings


def input_keys(objects: ObjectStore, prefix: str) -> list[str]:
    return [k for k in objects.list(prefix) if k.endswith(".dcm")]


def _dataset_bytes(ds: pydicom.Dataset) -> bytes:
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


def _check_cancel(ctx: RunContext, job_id: str) -> None:
    if ctx.store.is_cancel_requested(job_id):
        raise JobCancelledError


def _advance(ctx: RunContext, job_id: str, status: JobStatus) -> None:
    _check_cancel(ctx, job_id)
    ctx.store.transition(job_id, status)
    log.info("job stage", extra={"job_id": job_id, "status": status.value})


def build_manifest(
    *,
    job_id: str,
    input_manifest_sha256: str,
    settings: Settings,
    profile: PipelineProfile,
    model_info: dict[str, Any],
    outputs: list[dict[str, Any]],
    status: str,
    timings: dict[str, float],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "job_id": job_id,
        "input_manifest_sha256": input_manifest_sha256,
        "code_revision": settings.code_revision,
        "code_version": __version__,
        "container_digest": settings.container_digest,
        "model": {
            "name": model_info["name"],
            "version": model_info["version"],
            "weights_sha256": model_info["weights_sha256"],
            "synthetic": model_info["synthetic"],
        },
        "profile_sha256": profile.sha256,
        "preprocessing": {
            "config_version": profile.preprocessing.version,
            "config_sha256": profile.preprocessing.sha256,
        },
        "postprocessing": {
            "config_version": profile.postprocessing.version,
            "config_sha256": profile.postprocessing.sha256,
        },
        "runtime": {
            "framework": model_info["framework"],
            "device": settings.device,
            "cuda_version": None,
            "python": platform.python_version(),
            "libraries": {
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "pydicom": pydicom.__version__,
                "highdicom": highdicom.__version__,
            },
        },
        "outputs": [{"type": o["type"], "sha256": o["sha256"]} for o in outputs],
        "timings_seconds": timings,
        "status": status,
    }


def run_job(ctx: RunContext, job_id: str) -> None:
    """Run a job already claimed by this worker (status PREPROCESSING)."""
    try:
        _run(ctx, job_id)
    except JobCancelledError:
        ctx.store.transition(job_id, JobStatus.CANCELLED, detail={"reason": "user"})
        log.info("job cancelled", extra={"job_id": job_id})
    except PipelineError as exc:
        job = ctx.store.record_failure(job_id, exc, ctx.settings.retry_backoff_seconds)
        log.warning(
            "job stage failed",
            extra={"job_id": job_id, "error_code": exc.code, "status": job.status},
        )
    except MemoryError:
        ctx.store.record_failure(
            job_id,
            TransientError("Out of memory", code="OUT_OF_MEMORY"),
            ctx.settings.retry_backoff_seconds,
        )
    except Exception as exc:  # unexpected bugs: fail without leaking details to the API
        log.exception("unexpected job error", extra={"job_id": job_id})
        ctx.store.record_failure(
            job_id,
            PipelineError(f"Internal error ({type(exc).__name__})", code="INTERNAL_ERROR"),
            ctx.settings.retry_backoff_seconds,
        )


def _run(ctx: RunContext, job_id: str) -> None:
    timings: dict[str, float] = {}
    job = ctx.store.get(job_id)
    profile = load_profile(ctx.settings.configs_dir, job.model_name)

    t0 = time.perf_counter()
    paths = [ctx.objects.local_path(k) for k in input_keys(ctx.objects, job.input_prefix)]
    series = load_series(paths, IngestionOptions(series_instance_uid=job.series_instance_uid))
    assert series.volume_hu is not None
    pre = preprocess(series.volume_hu, series.geometry, profile.preprocessing)
    timings["preprocessing"] = time.perf_counter() - t0

    _advance(ctx, job_id, JobStatus.INFERENCE)
    t0 = time.perf_counter()
    adapter = create_adapter(profile.model.name)
    adapter.load(profile.model.artifact, ctx.settings.device, profile.model.params)
    params = {**profile.model.params, **job.params.get("model_params", {})}
    prediction = adapter.predict(pre.volume, pre.metadata, params)
    info = adapter.model_info()
    timings["inference"] = time.perf_counter() - t0

    _advance(ctx, job_id, JobStatus.POSTPROCESSING)
    t0 = time.perf_counter()
    result = postprocess(prediction, pre, profile.postprocessing)
    timings["postprocessing"] = time.perf_counter() - t0

    _advance(ctx, job_id, JobStatus.EXPORTING)
    t0 = time.perf_counter()
    export_ctx = ExportContext(model=info)
    seg = build_segmentation(series.datasets, result, export_ctx)
    sr = build_measurement_report(series.datasets, seg, result, export_ctx)
    overlay_png, overlay_meta = render_overlay(
        series.volume_hu,
        result,
        window_center=profile.overlay.window_center,
        window_width=profile.overlay.window_width,
        alpha=profile.overlay.alpha,
    )
    findings = {
        "synthetic": result.synthetic,
        "disclaimer": "Research prototype output. Not for diagnostic use.",
        "geometry": result.geometry.to_dict(),
        "findings": [f.to_dict() for f in result.findings],
        "qc_warnings": result.qc_warnings,
        "overlay": overlay_meta,
        "transforms": pre.transforms,
    }
    out_prefix = f"jobs/{job_id}/outputs"
    payloads = [
        ("DICOM_SEG", "seg.dcm", "application/dicom", _dataset_bytes(seg)),
        ("DICOM_SR", "sr.dcm", "application/dicom", _dataset_bytes(sr)),
        ("OVERLAY", "overlay.png", "image/png", overlay_png),
        ("FINDINGS", "findings.json", "application/json", json.dumps(findings, indent=2).encode()),
    ]
    artifacts = []
    for kind, name, content_type, data in payloads:
        key = f"{out_prefix}/{name}"
        digest = ctx.objects.put(key, data)
        artifacts.append(
            {
                "type": kind,
                "key": key,
                "sha256": digest,
                "size": len(data),
                "content_type": content_type,
            }
        )
    timings["exporting"] = time.perf_counter() - t0

    manifest = build_manifest(
        job_id=job_id,
        input_manifest_sha256=series.input_manifest_sha256,
        settings=ctx.settings,
        profile=profile,
        model_info=info.to_dict(),
        outputs=artifacts,
        status=JobStatus.SUCCEEDED.value,
        timings=timings,
    )
    manifest_bytes = json.dumps(manifest, indent=2).encode()
    manifest_key = f"{out_prefix}/manifest.json"
    ctx.objects.put(manifest_key, manifest_bytes)
    artifacts.append(
        {
            "type": "MANIFEST",
            "key": manifest_key,
            "sha256": sha256_bytes(manifest_bytes),
            "size": len(manifest_bytes),
            "content_type": "application/json",
        }
    )
    ctx.store.replace_artifacts(job_id, artifacts)
    _check_cancel(ctx, job_id)
    ctx.store.transition(
        job_id,
        JobStatus.SUCCEEDED,
        manifest=manifest,
        warnings=[*series.warnings, *result.qc_warnings],
        error_code=None,
        error_message=None,
    )
    log.info("job succeeded", extra={"job_id": job_id})
