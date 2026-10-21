# Roadmap (mapped to PROJECT_SPEC_FA.md section 22)

| Stage | Status in 0.1.0 |
|---|---|
| 1. Skeleton, config, schemas, logging, lint/type-check, README | Done |
| 2. DICOM ingest and validation, geometry tests | Done (single-frame CT; see dicom-support.md) |
| 3. Preprocessing (HU, window, normalisation, resampling) with versioned config | Done |
| 4. ModelAdapter, DummyAdapter, artifact hashing, postprocessing | Done (no real model yet) |
| 5. Queue and workers: lifecycle, idempotency, retry, dead letter, cancel, recovery | Done with a DB-backed queue; RabbitMQ not added |
| 6. DICOM SEG | Implemented and structurally tested; independent validation pending |
| 7. DICOM SR (TID 1500) | Implemented and structurally tested; independent validation pending |
| 8. GPU workers and batch | Deployment and per-GPU worker model only; no GPU adapter, OOM handling is generic (`MemoryError`/`TransientError`) |
| 9. Docker and Kubernetes | Dockerfile, compose, kustomize manifests; no Helm chart, no ingress/TLS |
| 10. Full validation | Not started: viewer testing, security review, performance on target hardware |

## Decisions that need a human

1. Target model(s) and their clinical task, label set and SNOMED codes.
2. Target viewer/PACS for SEG/SR acceptance testing.
3. Queue technology (keep DB queue vs. RabbitMQ) and object storage (S3/MinIO).
4. Data retention, de-identification and access-control policy.
5. GPU hardware and the memory budget per model.
