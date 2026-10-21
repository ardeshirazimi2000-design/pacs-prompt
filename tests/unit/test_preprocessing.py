from __future__ import annotations

import numpy as np
import pytest

from ctpipeline.common.errors import PreprocessingError
from ctpipeline.ingestion.geometry import VolumeGeometry
from ctpipeline.preprocessing.pipeline import (
    PreprocessingConfig,
    ResampleConfig,
    WindowConfig,
    labels_to_reference,
    preprocess,
)

GEOM = VolumeGeometry(
    origin=(10.0, 20.0, 30.0),
    spacing=(0.5, 0.5, 2.0),
    direction=tuple(np.eye(3).reshape(-1)),
    shape=(8, 16, 16),
)


def _cfg(**kw):
    return PreprocessingConfig(version="t1", window=WindowConfig(center=0, width=200), **kw)


def test_window_normalisation_range() -> None:
    vol = np.linspace(-500, 500, np.prod(GEOM.shape)).reshape(GEOM.shape).astype(np.float32)
    pre = preprocess(vol, GEOM, _cfg())
    assert pre.volume.min() == pytest.approx(0.0)
    assert pre.volume.max() == pytest.approx(1.0)
    assert pre.geometry == GEOM
    assert [t["op"] for t in pre.transforms] == ["window", "normalize"]


def test_config_hash_is_stable_and_sensitive() -> None:
    assert _cfg().sha256 == _cfg().sha256
    other = PreprocessingConfig(version="t1", window=WindowConfig(center=1, width=200))
    assert other.sha256 != _cfg().sha256


def test_resample_preserves_extent_and_shifts_origin() -> None:
    vol = np.zeros(GEOM.shape, np.float32)
    pre = preprocess(vol, GEOM, _cfg(resample=ResampleConfig(target_spacing=(1.0, 1.0, 2.0))))
    assert pre.volume.shape == (8, 8, 8)
    assert pre.spacing == pytest.approx((1.0, 1.0, 2.0))
    # Physical extent (shape * spacing) is unchanged.
    ext_in = np.asarray(GEOM.shape[::-1]) * np.asarray(GEOM.spacing)
    ext_out = np.asarray(pre.volume.shape[::-1]) * np.asarray(pre.spacing)
    np.testing.assert_allclose(ext_in, ext_out)
    # Voxel-edge alignment: new origin moves by half the spacing increase.
    assert pre.origin == pytest.approx((10.25, 20.25, 30.0))


def test_resample_keeps_object_position() -> None:
    vol = np.zeros(GEOM.shape, np.float32)
    vol[3:5, 6:10, 4:8] = 100.0
    pre = preprocess(
        vol, GEOM, _cfg(resample=ResampleConfig(target_spacing=(1.0, 1.0, 2.0), order=0))
    )

    def centroid_mm(v, g):
        idx = np.argwhere(v > 0.5 * v.max()).mean(axis=0)
        return g.index_to_physical(idx)

    np.testing.assert_allclose(
        centroid_mm(pre.volume, pre.geometry), centroid_mm(vol, GEOM), atol=1e-6
    )


def test_labels_round_trip_to_reference() -> None:
    vol = np.zeros(GEOM.shape, np.float32)
    pre = preprocess(vol, GEOM, _cfg(resample=ResampleConfig(target_spacing=(1.0, 1.0, 2.0))))
    labels = np.zeros(pre.volume.shape, np.uint8)
    labels[2, 3, 4] = 1
    back = labels_to_reference(labels, pre)
    assert back.shape == GEOM.shape
    assert set(np.unique(back)) == {0, 1}  # nearest neighbour, never interpolated
    np.testing.assert_array_equal(np.argwhere(back), [[2, 6, 8], [2, 6, 9], [2, 7, 8], [2, 7, 9]])


def test_rejects_bad_input() -> None:
    with pytest.raises(PreprocessingError):
        preprocess(np.zeros((2, 2, 2), np.float32), GEOM, _cfg())
    vol = np.zeros(GEOM.shape, np.float32)
    vol[0, 0, 0] = np.nan
    with pytest.raises(PreprocessingError):
        preprocess(vol, GEOM, _cfg())
    with pytest.raises(PreprocessingError):
        preprocess(np.zeros(GEOM.shape, np.float32), GEOM, PreprocessingConfig(version="x"))
