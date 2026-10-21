"""Structured error types.

Error messages must never contain patient-identifying information (PHI). They
may contain UIDs only when explicitly needed for traceability.
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for all pipeline errors.

    ``retryable`` tells the job runner whether a retry could succeed.
    """

    code = "PIPELINE_ERROR"
    retryable = False

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code
        self.message = message

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


class IngestionError(PipelineError):
    code = "INGESTION_ERROR"


class PreprocessingError(PipelineError):
    code = "PREPROCESSING_ERROR"


class ModelError(PipelineError):
    code = "MODEL_ERROR"


class TransientError(PipelineError):
    """An error that may succeed on retry (I/O hiccup, OOM, worker loss)."""

    code = "TRANSIENT_ERROR"
    retryable = True


class QualityCheckError(PipelineError):
    """A QC check failed and the output must not be published."""

    code = "QC_BLOCKING_ERROR"


class ExportError(PipelineError):
    code = "EXPORT_ERROR"


class InvalidTransitionError(PipelineError):
    code = "INVALID_STATE_TRANSITION"
