# Operations

## Processes

* **API** (`uvicorn ctpipeline.api.app:build_app --factory`): stores uploads, validates
  them synchronously, enqueues jobs, serves status/results. No model code runs here.
* **Worker** (`ctpipeline-worker`): polls the `jobs` table, claims one job at a time with a
  conditional `UPDATE ... WHERE status='QUEUED'`, runs it, writes artifacts and manifest.
  `SIGTERM` finishes the current job, then exits (set `terminationGracePeriodSeconds`).
* GPU: run one worker pod per GPU (`nvidia.com/gpu: 1`, `CTP_DEVICE=cuda:0`). GPU count is
  not enough to size concurrency: model memory and input size must be measured first.

## Retry, dead letter, recovery

* Errors are classified: `TransientError` (e.g. `OUT_OF_MEMORY`, `WORKER_LOST`) is retried
  up to `CTP_MAX_ATTEMPTS` with exponential backoff (`CTP_RETRY_BACKOFF_SECONDS * 2^n`);
  when exhausted the job becomes `FAILED` with `dead_lettered = true`.
* Validation, model and QC errors fail immediately (`dead_lettered = false`).
* Workers periodically requeue jobs whose heartbeat is older than
  `CTP_STALE_JOB_SECONDS` (worker crash/restart recovery). Outputs are written under
  deterministic keys, so a re-run overwrites partial outputs.

## Database

SQLite is the default for development; set `CTP_DATABASE_URL` to PostgreSQL
(`postgresql+psycopg://...`) for anything shared. `init_db()` only creates missing tables.
Before production, introduce Alembic migrations and run them as a separate, controlled step.

## Kubernetes

`deploy/k8s` (kustomize): namespace, ConfigMap, RWX PVC for objects, API Deployment +
Service with readiness/liveness probes, CPU and GPU worker Deployments, NetworkPolicies.
Containers run as UID 10001, read-only root filesystem, all capabilities dropped.
Secrets (`ctpipeline-secrets`: `CTP_DATABASE_URL`, `CTP_API_KEY`) are created out of band.
Rollback: `kubectl -n ctpipeline rollout undo deployment/<name>`; pin images by digest.
Autoscaling should use queue depth (`ctpipeline_queue_depth` on `/metrics`), not CPU.
TLS terminates at the ingress (not included).

## Privacy

* Logs are JSON and never include patient attributes; a filter redacts well-known PHI keys
  as a second line of defence. A test asserts patient name/ID never reach the logs.
* Error messages returned by the API never contain patient attributes.
* Output file names use the job id only.
* Uploaded inputs are retained in object storage; a retention/deletion policy and
  de-identification policy must be defined for the target environment.
