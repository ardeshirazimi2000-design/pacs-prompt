"""Structural checks of SEG/SR outputs.

These tests re-read the written files and verify references and geometry.
They are not a substitute for an independent IOD validator (see CI job
``dicom-conformance`` and docs/dicom-support.md).
"""

from __future__ import annotations

import io
from pathlib import Path

import highdicom as hd
import numpy as np
import pydicom
import pytest
from PIL import Image

from ctpipeline.adapters.base import ModelInfo, SegmentClass
from ctpipeline.adapters.dummy import TISSUE
from ctpipeline.common.errors import ExportError
from ctpipeline.ingestion.loader import IngestedSeries, load_series
from ctpipeline.outputs.common import ExportContext
from ctpipeline.outputs.dicom_seg import build_segmentation
from ctpipeline.outputs.dicom_sr import build_measurement_report
from ctpipeline.outputs.overlay import render_overlay
from ctpipeline.postprocessing.pipeline import PostprocessedResult
from ctpipeline.testing.synthetic import make_ct_series

INFO = ModelInfo(
    name="unit-test",
    version="1.0",
    task="segmentation",
    weights_sha256="0" * 64,
    framework="none",
    synthetic=True,
)
CTX = ExportContext(model=INFO)


def _roundtrip(ds: pydicom.Dataset) -> pydicom.Dataset:
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    buf.seek(0)
    return pydicom.dcmread(buf)


@pytest.fixture
def series(tmp_path: Path) -> IngestedSeries:
    angle = np.deg2rad(20)
    return load_series(
        make_ct_series(
            tmp_path,
            row_cosine=(np.cos(angle), np.sin(angle), 0),
            col_cosine=(-np.sin(angle), np.cos(angle), 0),
        )
    )


def _result(series: IngestedSeries, label_map: np.ndarray, n_classes: int = 1):
    classes = [
        SegmentClass(i, f"SYNTHETIC class {i}", TISSUE, TISSUE) for i in range(1, n_classes + 1)
    ]
    return PostprocessedResult(
        label_map=label_map, classes=classes, findings=[], geometry=series.geometry, synthetic=True
    )


def test_seg_references_and_geometry(series: IngestedSeries) -> None:
    labels = np.zeros(series.geometry.shape, np.uint8)
    labels[4:7, 10:20, 12:18] = 1
    seg_ds = _roundtrip(build_segmentation(series.datasets, _result(series, labels), CTX))
    seg = hd.seg.Segmentation.from_dataset(seg_ds)

    assert seg_ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.66.4"
    assert seg_ds.FrameOfReferenceUID == series.frame_of_reference_uid
    assert seg_ds.StudyInstanceUID == series.study_instance_uid
    assert seg_ds.SegmentSequence[0].SegmentLabel.startswith("SYNTHETIC")
    assert seg_ds.SegmentSequence[0].SegmentAlgorithmType == "AUTOMATIC"
    referenced = {
        item.ReferencedSOPInstanceUID
        for s in seg_ds.ReferencedSeriesSequence
        for item in s.ReferencedInstanceSequence
    }
    assert referenced == set(series.sop_instance_uids)

    # Every frame's ImagePositionPatient equals its source slice position, and
    # the mask in that frame equals the input mask slice.
    by_sop = {str(ds.SOPInstanceUID): k for k, ds in enumerate(series.datasets)}
    pixels = seg_ds.pixel_array
    for f, frame in enumerate(seg_ds.PerFrameFunctionalGroupsSequence):
        src_uid = frame.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID
        k = by_sop[src_uid]
        np.testing.assert_allclose(
            [float(v) for v in frame.PlanePositionSequence[0].ImagePositionPatient],
            [float(v) for v in series.datasets[k].ImagePositionPatient],
            atol=1e-4,
        )
        np.testing.assert_array_equal(pixels[f].astype(bool), labels[k] == 1)
    assert seg.number_of_segments == 1


def test_seg_empty_and_multi_segment(series: IngestedSeries) -> None:
    empty = np.zeros(series.geometry.shape, np.uint8)
    ds = _roundtrip(build_segmentation(series.datasets, _result(series, empty), CTX))
    assert len(ds.SegmentSequence) == 1

    labels = np.zeros(series.geometry.shape, np.uint8)
    labels[2, 5:9, 5:9] = 1
    labels[8, 20:25, 20:25] = 2
    ds = _roundtrip(build_segmentation(series.datasets, _result(series, labels, 2), CTX))
    assert [s.SegmentNumber for s in ds.SegmentSequence] == [1, 2]
    seg = hd.seg.Segmentation.from_dataset(ds)
    sop_uids = [(str(series.series_instance_uid), uid) for uid in series.sop_instance_uids]
    vol = seg.get_pixels_by_source_instance(
        source_sop_instance_uids=[u for _, u in sop_uids],
        combine_segments=True,
    )
    np.testing.assert_array_equal(vol, labels)


def test_seg_geometry_mismatch_rejected(series: IngestedSeries) -> None:
    bad = np.zeros((3, 4, 4), np.uint8)
    with pytest.raises(ExportError):
        build_segmentation(series.datasets, _result(series, bad), CTX)


def test_sr_references_seg(series: IngestedSeries) -> None:
    labels = np.zeros(series.geometry.shape, np.uint8)
    labels[4:6, 10:12, 10:12] = 1
    res = _result(series, labels)
    seg = build_segmentation(series.datasets, res, CTX)
    sr_ds = _roundtrip(build_measurement_report(series.datasets, seg, res, CTX))
    sr = hd.sr.Comprehensive3DSR.from_dataset(sr_ds)
    assert sr_ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.88.34"
    assert sr_ds.ContentTemplateSequence[0].TemplateIdentifier == "1500"
    evidence = {
        i.ReferencedSOPInstanceUID
        for st in sr_ds.CurrentRequestedProcedureEvidenceSequence
        for se in st.ReferencedSeriesSequence
        for i in se.ReferencedSOPSequence
    }
    assert seg.SOPInstanceUID in evidence
    groups = sr.content.get_volumetric_roi_measurement_groups()
    assert len(groups) == 1
    ref = groups[0].referenced_segment
    assert ref is not None
    assert ref.referenced_sop_instance_uid == seg.SOPInstanceUID
    assert ref.referenced_segment_numbers == [1]
    measurement = groups[0].get_measurements()[0]
    voxel = float(np.prod(series.geometry.spacing))
    assert measurement.value == pytest.approx(8 * voxel)
    assert measurement.unit.value == "mm3"


def test_overlay_png(series: IngestedSeries) -> None:
    labels = np.zeros(series.geometry.shape, np.uint8)
    labels[7, 5:9, 5:9] = 1
    assert series.volume_hu is not None
    png, meta = render_overlay(series.volume_hu, _result(series, labels))
    img = Image.open(io.BytesIO(png))
    assert img.format == "PNG"
    assert img.size == (32, 32 + 14)
    assert meta["slice_index"] == 7
    assert meta["synthetic"] is True
    with pytest.raises(ValueError):
        render_overlay(series.volume_hu[:2], _result(series, labels))
