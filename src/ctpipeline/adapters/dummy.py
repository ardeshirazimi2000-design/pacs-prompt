"""Threshold-based dummy adapter for infrastructure testing ONLY.

Its output is synthetic and has NO medical meaning. Every prediction it
produces is flagged ``synthetic=True`` and exported with a "SYNTHETIC" label.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ctpipeline.adapters.base import (
    CodedConcept,
    ModelAdapter,
    ModelInfo,
    SegmentationPrediction,
    SegmentClass,
)
from ctpipeline.common.errors import ModelError
from ctpipeline.common.hashing import sha256_file, sha256_json

# SNOMED CT 85756007 "Tissue" is used for both category and type: a real,
# generic concept rather than an invented code.
TISSUE = CodedConcept(value="85756007", scheme="SCT", meaning="Tissue")


class DummyThresholdAdapter(ModelAdapter):
    name = "dummy-threshold"
    version = "0.1.0"

    def __init__(self) -> None:
        self._loaded = False
        self._config: dict[str, Any] = {}
        self._weights_sha256 = ""
        self._device = "cpu"

    def load(self, model_artifact: str | None, device: str, config: dict[str, Any]) -> None:
        if device != "cpu":
            raise ModelError("DummyThresholdAdapter only runs on CPU", code="UNSUPPORTED_DEVICE")
        self._config = {"threshold": 0.5, **config}
        # There are no weights; hash the artifact if provided, otherwise the config.
        self._weights_sha256 = (
            sha256_file(model_artifact) if model_artifact else sha256_json(self._config)
        )
        self._device = device
        self._loaded = True

    def predict(
        self, volume: np.ndarray, metadata: dict[str, Any], config: dict[str, Any]
    ) -> SegmentationPrediction:
        if not self._loaded:
            raise ModelError("Model is not loaded", code="MODEL_NOT_LOADED")
        if volume.ndim != 3:
            raise ModelError("Expected a 3D volume", code="INVALID_MODEL_INPUT")
        threshold = float(config.get("threshold", self._config["threshold"]))
        label_map = (volume >= threshold).astype(np.uint8)
        return SegmentationPrediction(
            label_map=label_map,
            classes=[
                SegmentClass(
                    label=1,
                    name="SYNTHETIC dummy region",
                    category=TISSUE,
                    property_type=TISSUE,
                )
            ],
            synthetic=True,
            extra={"threshold": threshold},
        )

    def health_check(self) -> bool:
        return self._loaded

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            version=self.version,
            task="segmentation",
            weights_sha256=self._weights_sha256,
            framework="numpy-" + np.__version__,
            synthetic=True,
            description="Intensity threshold for pipeline testing. Not a medical model.",
        )
