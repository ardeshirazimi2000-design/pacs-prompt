from __future__ import annotations

import json
from datetime import timedelta
from unittest import mock

import highdicom as hd
import pydicom
import pytest
from fastapi.testclient import TestClient

from ctpipeline.api.app import create_app
from ctpipeline.common.errors import TransientError
from ctpipeline.jobs.db import utcnow
from ctpipeline.jobs.runner import RunContext
from ctpipeline.jobs.states import JobStatus
from ctpipeline.jobs.worker import Worker


@pytest.fixture
def client(ctx: RunContext) -> TestClient:
    return TestClient(create_app(ctx))


def _upload(client: TestClient, payloads: list[bytes], **headers: str):
    files = [("files", (f"{i}.dcm", data, "application/dicom")) for i, data in enumerate(payloads)]
    return client.post("/v1/studies", files=files, headers=headers)


def test_health(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").status_code == 200


def test_end_to_end_success(client: TestClient, ctx: RunContext, series_bytes) -> None:
    resp = _upload(client, series_bytes)
    assert resp.status_code == 202, resp.text
    job = resp.json()
    assert job["status"] == "QUEUED"
    assert client.get(f"/v1/jobs/{job['job_id']}/results").status_code == 409

    assert Worker(ctx, "w1").run_once()
    status = client.get(f"/v1/jobs/{job['job_id']}").json()
    assert status["status"] == "SUCCEEDED", status

    results = client.get(f"/v1/jobs/{job['job_id']}/results").json()
    types = {a["type"] for a in results["artifacts"]}
    assert types == {"DICOM_SEG", "DICOM_SR", "OVERLAY", "FINDINGS", "MANIFEST"}
    manifest = results["manifest"]
    assert manifest["model"]["synthetic"] is True
    assert manifest["input_manifest_sha256"]
    assert {o["type"] for o in manifest["outputs"]} == {
        "DICOM_SEG",
        "DICOM_SR",
        "OVERLAY",
        "FINDINGS",
    }

    seg = client.get(f"/v1/jobs/{job['job_id']}/results/DICOM_SEG")
    assert seg.status_code == 200
    import io

    seg_ds = pydicom.dcmread(io.BytesIO(seg.content))
    hd.seg.Segmentation.from_dataset(seg_ds)
    assert seg_ds.ContentDescription.startswith("SYNTHETIC")
    findings = json.loads(client.get(f"/v1/jobs/{job['job_id']}/results/findings").content)
    assert findings["synthetic"] is True
    assert len(findings["findings"]) == 1

    events = [e["to_status"] for e in client.get(f"/v1/jobs/{job['job_id']}/events").json()]
    assert events == [
        "RECEIVED",
        "VALIDATING",
        "QUEUED",
        "PREPROCESSING",
        "INFERENCE",
        "POSTPROCESSING",
        "EXPORTING",
        "SUCCEEDED",
    ]
    assert 'ctpipeline_jobs{status="SUCCEEDED"} 1' in client.get("/metrics").text


def test_idempotency_key(client: TestClient, series_bytes) -> None:
    a = _upload(client, series_bytes, **{"Idempotency-Key": "abc-1"})
    b = _upload(client, series_bytes, **{"Idempotency-Key": "abc-1"})
    assert a.status_code == 202
    assert b.status_code == 200
    assert a.json()["job_id"] == b.json()["job_id"]


def test_invalid_input_fails_with_code(client: TestClient) -> None:
    resp = _upload(client, [b"garbage"])
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "FAILED"
    assert body["error"]["code"] == "INVALID_DICOM_FILE"


def test_unknown_model_rejected(client: TestClient, series_bytes) -> None:
    files = [("files", ("0.dcm", series_bytes[0], "application/dicom"))]
    resp = client.post("/v1/studies", files=files, data={"model": "../etc"})
    assert resp.status_code == 422


def test_cancel_queued_and_running(client: TestClient, ctx: RunContext, series_bytes) -> None:
    job_id = _upload(client, series_bytes).json()["job_id"]
    assert client.post(f"/v1/jobs/{job_id}/cancel").json()["status"] == "CANCELLED"
    assert client.post(f"/v1/jobs/{job_id}/cancel").status_code == 409
    assert not Worker(ctx).run_once()  # cancelled jobs are never claimed

    job_id = _upload(client, series_bytes).json()["job_id"]
    real_predict = None

    def cancel_during_inference(self, *args, **kwargs):
        ctx.store.request_cancel(job_id)
        return real_predict(self, *args, **kwargs)

    from ctpipeline.adapters.dummy import DummyThresholdAdapter

    real_predict = DummyThresholdAdapter.predict
    with mock.patch.object(DummyThresholdAdapter, "predict", cancel_during_inference):
        Worker(ctx).run_once()
    assert ctx.store.get(job_id).status == JobStatus.CANCELLED.value


def test_retry_then_dead_letter(client: TestClient, ctx: RunContext, series_bytes) -> None:
    job_id = _upload(client, series_bytes).json()["job_id"]
    boom = TransientError("simulated GPU OOM", code="OUT_OF_MEMORY")
    with mock.patch("ctpipeline.jobs.runner.preprocess", side_effect=boom):
        worker = Worker(ctx)
        for attempt in range(1, 4):
            assert worker.run_once()
            job = ctx.store.get(job_id)
            assert job.attempts == attempt
        assert not worker.run_once()
    job = ctx.store.get(job_id)
    assert job.status == JobStatus.FAILED.value
    assert job.dead_lettered is True
    assert job.error_code == "OUT_OF_MEMORY"


def test_non_retryable_error_fails_immediately(ctx: RunContext, client, series_bytes) -> None:
    from ctpipeline.common.errors import ModelError

    job_id = _upload(client, series_bytes).json()["job_id"]
    with mock.patch("ctpipeline.jobs.runner.preprocess", side_effect=ModelError("bad")):
        Worker(ctx).run_once()
    job = ctx.store.get(job_id)
    assert job.status == JobStatus.FAILED.value
    assert job.attempts == 1
    assert not job.dead_lettered


def test_recovery_after_worker_crash(client: TestClient, ctx: RunContext, series_bytes) -> None:
    job_id = _upload(client, series_bytes).json()["job_id"]
    claimed = ctx.store.claim_next("crashed-worker")
    assert claimed is not None and claimed.id == job_id
    # Nothing heart-beats; pretend the worker died long ago.
    ctx.store.set_fields(job_id, heartbeat_at=utcnow() - timedelta(hours=1))
    assert ctx.store.recover_stale(stale_after_seconds=60) == [job_id]
    assert ctx.store.get(job_id).status == JobStatus.QUEUED.value
    assert Worker(ctx, "w2").run_once()
    job = ctx.store.get(job_id)
    assert job.status == JobStatus.SUCCEEDED.value
    assert job.attempts == 2


def test_claim_is_exclusive(client: TestClient, ctx: RunContext, series_bytes) -> None:
    _upload(client, series_bytes)
    assert ctx.store.claim_next("a") is not None
    assert ctx.store.claim_next("b") is None


def test_priority_order(client: TestClient, ctx: RunContext, series_bytes) -> None:
    files = [("files", (f"{i}.dcm", d, "application/dicom")) for i, d in enumerate(series_bytes)]
    low = client.post("/v1/studies", files=files, data={"priority": "0"}).json()["job_id"]
    high = client.post("/v1/studies", files=files, data={"priority": "5"}).json()["job_id"]
    assert ctx.store.claim_next("w").id == high
    assert ctx.store.claim_next("w").id == low


def test_api_key_required_when_configured(ctx: RunContext, series_bytes) -> None:
    ctx.settings.api_key = "s3cret"
    client = TestClient(create_app(ctx))
    assert _upload(client, series_bytes).status_code == 401
    assert _upload(client, series_bytes, **{"X-API-Key": "s3cret"}).status_code == 202
    assert client.get("/healthz").status_code == 200


def test_logs_contain_no_phi(client: TestClient, ctx: RunContext, series_bytes, capsys) -> None:
    from ctpipeline.common.logging import configure_logging

    configure_logging("DEBUG")
    _upload(client, series_bytes)
    Worker(ctx).run_once()
    out = capsys.readouterr().out
    assert "job succeeded" in out
    assert "SYNTHETIC^PHANTOM" not in out
    assert "SYNTH0001" not in out
