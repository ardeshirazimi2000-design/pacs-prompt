from __future__ import annotations

from enum import StrEnum

from ctpipeline.common.errors import InvalidTransitionError


class JobStatus(StrEnum):
    RECEIVED = "RECEIVED"
    VALIDATING = "VALIDATING"
    QUEUED = "QUEUED"
    PREPROCESSING = "PREPROCESSING"
    INFERENCE = "INFERENCE"
    POSTPROCESSING = "POSTPROCESSING"
    EXPORTING = "EXPORTING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL = frozenset({JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED})
RUNNING = frozenset(
    {JobStatus.PREPROCESSING, JobStatus.INFERENCE, JobStatus.POSTPROCESSING, JobStatus.EXPORTING}
)

_STOP = {JobStatus.FAILED, JobStatus.CANCELLED}
# Running stages may go back to QUEUED for a bounded retry or crash recovery.
ALLOWED_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.RECEIVED: frozenset({JobStatus.VALIDATING, *_STOP}),
    JobStatus.VALIDATING: frozenset({JobStatus.QUEUED, *_STOP}),
    JobStatus.QUEUED: frozenset({JobStatus.PREPROCESSING, *_STOP}),
    JobStatus.PREPROCESSING: frozenset({JobStatus.INFERENCE, JobStatus.QUEUED, *_STOP}),
    JobStatus.INFERENCE: frozenset({JobStatus.POSTPROCESSING, JobStatus.QUEUED, *_STOP}),
    JobStatus.POSTPROCESSING: frozenset({JobStatus.EXPORTING, JobStatus.QUEUED, *_STOP}),
    JobStatus.EXPORTING: frozenset({JobStatus.SUCCEEDED, JobStatus.QUEUED, *_STOP}),
    JobStatus.SUCCEEDED: frozenset(),
    JobStatus.FAILED: frozenset(),
    JobStatus.CANCELLED: frozenset(),
}


def check_transition(current: JobStatus | str, target: JobStatus | str) -> None:
    cur, tgt = JobStatus(current), JobStatus(target)
    if tgt not in ALLOWED_TRANSITIONS[cur]:
        raise InvalidTransitionError(f"Transition {cur} -> {tgt} is not allowed")
