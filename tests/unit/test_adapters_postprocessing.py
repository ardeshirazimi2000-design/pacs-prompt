from __future__ import annotations

import numpy as np
import pytest

from ctpipeline.adapters.base import ModelAdapter, SegmentationPrediction, SegmentClass
from ctpipeline.adapters.dummy import TISSUE, DummyThresholdAdapter
from ctpipeline.adapters.registry import available_models, create_adapter
from ctpipeline.common.errors import ModelError, QualityCheckError
from ctpipeline.ingestion.geometry import VolumeGeometry
from ctpipeline.postprocessing.pipeline import PostprocessingConfig, postprocess
from ctpipeline.preprocessing.pipeline import PreprocessingConfig, preprocess

GEOM = VolumeGeometry(
    origin=(-50.0, -60.0, 100.0),
    spacing=(1.0, 2.0, 3.0),
    direction=tuple(np.eye(3).reshape(-1)),
    shape=(6, 10, 10),
)


def _pre():
    return preprocess(
        np.zeros(GEOM.shape, np.float32),
        GEOM,
        PreprocessingConfig(version="t", normalization="none"),
    )


def test_registry_and_contract() -> None:
    assert "dummy-threshold" in available_models()
    adapter = create_adapter("dummy-threshold")
    assert isinstance(adapter, ModelAdapter)
    assert not adapter.health_check()
    with pytest.raises(ModelError):
        adapter.predict(np.zeros((2, 2, 2)), {}, {})
    adapter.load(None, "cpu", {"threshold": 0.5})
    assert adapter.health_check()
    info = adapter.model_info()
    assert info.synthetic is True
    assert len(info.weights_sha256) == 64
    with pytest.raises(ModelError):
        create_adapter("nope")
    with pytest.raises(ModelError):
        DummyThresholdAdapter().load(None, "cuda:0", {})


def test_dummy_prediction_is_flagged_synthetic() -> None:
    adapter = DummyThresholdAdapter()
    adapter.load(None, "cpu", {})
    pred = adapter.predict(np.full((2, 3, 3), 0.7, np.float32), {}, {"threshold": 0.5})
    assert pred.synthetic
    assert pred.label_map.all()
    assert "SYNTHETIC" in pred.classes[0].name


def _prediction(label_map: np.ndarray) -> SegmentationPrediction:
    return SegmentationPrediction(
        label_map=label_map,
        classes=[SegmentClass(1, "a", TISSUE, TISSUE), SegmentClass(2, "b", TISSUE, TISSUE)],
        synthetic=True,
    )


def test_findings_physical_vs_voxel_coordinates() -> None:
    labels = np.zeros(GEOM.shape, np.uint8)
    labels[2:4, 4:6, 2:4] = 1  # 8 voxels
    labels[0, 0, 0] = 2  # tiny component
    res = postprocess(_prediction(labels), _pre(), PostprocessingConfig())
    big = next(f for f in res.findings if f.label == 1)
    assert big.voxel_count == 8
    assert big.volume_mm3 == pytest.approx(8 * 1 * 2 * 3)
    assert big.centroid_voxel == pytest.approx((2.5, 4.5, 2.5))
    # x = -50 + 2.5*1, y = -60 + 4.5*2, z = 100 + 2.5*3
    assert big.centroid_mm == pytest.approx((-47.5, -51.0, 107.5))


def test_small_component_removal_and_empty_qc() -> None:
    labels = np.zeros(GEOM.shape, np.uint8)
    labels[2:4, 4:6, 2:4] = 1
    labels[0, 0, 0] = 2
    res = postprocess(_prediction(labels), _pre(), PostprocessingConfig(min_component_voxels=2))
    assert (res.label_map == 2).sum() == 0
    assert any("'b' is empty" in w for w in res.qc_warnings)
    with pytest.raises(QualityCheckError):
        postprocess(
            _prediction(np.zeros(GEOM.shape, np.uint8)),
            _pre(),
            PostprocessingConfig(fail_on_empty=True),
        )


def test_undeclared_labels_block_output() -> None:
    labels = np.zeros(GEOM.shape, np.uint8)
    labels[1, 1, 1] = 7
    with pytest.raises(QualityCheckError):
        postprocess(_prediction(labels), _pre(), PostprocessingConfig())
