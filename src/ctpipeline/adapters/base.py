"""Common model adapter contract.

The orchestration code only talks to :class:`ModelAdapter`; swapping models
means registering a different adapter, never editing the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class CodedConcept:
    """A coded concept from an established coding scheme (e.g. SNOMED CT ``SCT``)."""

    value: str
    scheme: str
    meaning: str


@dataclass(frozen=True)
class SegmentClass:
    label: int
    name: str
    category: CodedConcept
    property_type: CodedConcept


@dataclass
class SegmentationPrediction:
    """Model output in *model space* (the preprocessed grid)."""

    label_map: np.ndarray  # (k, j, i) integer labels, 0 = background
    classes: list[SegmentClass]
    synthetic: bool
    probabilities: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    task: Literal["segmentation"] = "segmentation"


@dataclass(frozen=True)
class ModelInfo:
    name: str
    version: str
    task: str
    weights_sha256: str
    framework: str
    synthetic: bool
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "task": self.task,
            "weights_sha256": self.weights_sha256,
            "framework": self.framework,
            "synthetic": self.synthetic,
            "description": self.description,
        }


class ModelAdapter(ABC):
    @abstractmethod
    def load(self, model_artifact: str | None, device: str, config: dict[str, Any]) -> None:
        """Load model weights and runtime configuration."""

    @abstractmethod
    def predict(
        self, volume: np.ndarray, metadata: dict[str, Any], config: dict[str, Any]
    ) -> SegmentationPrediction:
        """Return standardized model predictions in model space."""

    @abstractmethod
    def health_check(self) -> bool: ...

    @abstractmethod
    def model_info(self) -> ModelInfo: ...
