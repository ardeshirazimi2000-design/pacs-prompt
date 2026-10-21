from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.encaps import encapsulate
from pydicom.uid import JPEGBaseline8Bit

from ctpipeline.common.errors import IngestionError
from ctpipeline.ingestion.loader import IngestionOptions, load_series
from ctpipeline.testing.synthetic import make_ct_series, make_phantom


def _codes(paths, **kw):
    with pytest.raises(IngestionError) as info:
        load_series(paths, IngestionOptions(**kw))
    return info.value.code


def test_sorts_by_geometry_not_filename(tmp_path: Path) -> None:
    vol = make_phantom()
    paths = make_ct_series(tmp_path, vol, shuffle_names=True)
    series = load_series(sorted(paths))  # file name order is a permutation
    assert series.volume_hu is not None
    np.testing.assert_allclose(series.volume_hu, np.round(vol), atol=0.5)
    z = [float(ds.ImagePositionPatient[2]) for ds in series.datasets]
    assert z == sorted(z)


def test_geometry_axial(tmp_path: Path) -> None:
    paths = make_ct_series(tmp_path, spacing=(0.7, 0.9, 3.0), origin=(1.0, 2.0, 3.0))
    g = load_series(paths).geometry
    assert g.spacing == pytest.approx((0.7, 0.9, 3.0))
    assert g.origin == pytest.approx((1.0, 2.0, 3.0))
    np.testing.assert_allclose(g.direction_matrix, np.eye(3), atol=1e-6)
    assert g.shape == (12, 32, 32)


def test_geometry_reversed_and_oblique(tmp_path: Path) -> None:
    # Rows along +x, columns along -z rotated: an oblique, non-axial stack.
    angle = np.deg2rad(30)
    row = (np.cos(angle), np.sin(angle), 0.0)
    col = (0.0, 0.0, -1.0)
    paths = make_ct_series(tmp_path, row_cosine=row, col_cosine=col, spacing=(1.0, 1.0, 2.0))
    series = load_series(paths)
    g = series.geometry
    normal = np.cross(row, col)
    np.testing.assert_allclose(g.direction_matrix[:, 2], normal, atol=1e-5)
    # The k-th slice must sit at origin + k * spacing * normal.
    for k, ds in enumerate(series.datasets):
        expected = np.asarray(g.origin) + normal * 2.0 * k
        np.testing.assert_allclose(np.asarray(ds.ImagePositionPatient, float), expected, atol=1e-4)
    # Voxel/physical round trip.
    p = g.index_to_physical((3, 4, 5))
    np.testing.assert_allclose(g.physical_to_index(p), (3, 4, 5), atol=1e-9)


def test_rescale_slope_intercept(tmp_path: Path) -> None:
    vol = make_phantom()
    paths = make_ct_series(tmp_path, vol, slope=2.0, intercept=-2048.0)
    series = load_series(paths)
    assert series.volume_hu is not None
    np.testing.assert_allclose(series.volume_hu, vol, atol=1.01)


def test_missing_slice_rejected(tmp_path: Path) -> None:
    paths = sorted(make_ct_series(tmp_path, shuffle_names=False))
    del paths[5]
    assert _codes(paths) == "NON_UNIFORM_SLICE_SPACING"


def test_corrupt_file_rejected(tmp_path: Path) -> None:
    paths = make_ct_series(tmp_path)
    bad = tmp_path / "bad.dcm"
    bad.write_bytes(b"not a dicom file")
    assert _codes([*paths, bad]) == "INVALID_DICOM_FILE"


def test_identical_duplicate_ignored(tmp_path: Path) -> None:
    paths = make_ct_series(tmp_path / "a")
    dup = tmp_path / "dup.dcm"
    dup.write_bytes(paths[0].read_bytes())
    series = load_series([*paths, dup])
    assert len(series.datasets) == len(paths)
    assert series.warnings


def test_conflicting_duplicate_rejected(tmp_path: Path) -> None:
    paths = make_ct_series(tmp_path / "a")
    ds = pydicom.dcmread(paths[0])
    ds.SliceThickness = "9"
    other = tmp_path / "other.dcm"
    ds.save_as(other)
    assert _codes([*paths, other]) == "DUPLICATE_SOP_INSTANCE_CONFLICT"


def test_multiple_series_requires_selection(tmp_path: Path) -> None:
    a = make_ct_series(tmp_path / "a", study_uid="1.2.3.4", seed=1)
    b = make_ct_series(tmp_path / "b", study_uid="1.2.3.4", seed=2)
    assert _codes([*a, *b]) == "MULTIPLE_SERIES"
    wanted = str(pydicom.dcmread(b[0]).SeriesInstanceUID)
    series = load_series([*a, *b], IngestionOptions(series_instance_uid=wanted))
    assert series.series_instance_uid == wanted
    assert _codes([*a, *b], series_instance_uid="9.9.9") == "SERIES_NOT_FOUND"


def test_wrong_modality_rejected(tmp_path: Path) -> None:
    paths = make_ct_series(tmp_path)
    ds = pydicom.dcmread(paths[0])
    ds.Modality = "MR"
    ds.save_as(paths[0])
    assert _codes(paths) == "UNSUPPORTED_MODALITY"


def test_unsupported_transfer_syntax(tmp_path: Path) -> None:
    paths = make_ct_series(tmp_path)
    ds = pydicom.dcmread(paths[0])
    ds.file_meta.TransferSyntaxUID = JPEGBaseline8Bit
    ds.PixelData = encapsulate([b"\xff\xd8fakejpeg\xff\xd9"])
    ds.save_as(paths[0], enforce_file_format=True)
    assert _codes(paths) == "UNSUPPORTED_TRANSFER_SYNTAX"


def test_gantry_tilt_rejected(tmp_path: Path) -> None:
    paths = sorted(make_ct_series(tmp_path, shuffle_names=False))
    for k, p in enumerate(paths):
        ds = pydicom.dcmread(p)
        ipp = [float(v) for v in ds.ImagePositionPatient]
        ds.ImagePositionPatient = [ipp[0], ipp[1] + 0.5 * k, ipp[2]]
        ds.save_as(p)
    assert _codes(paths) == "UNSUPPORTED_GANTRY_TILT"


def test_inconsistent_frame_of_reference(tmp_path: Path) -> None:
    paths = make_ct_series(tmp_path)
    ds = pydicom.dcmread(paths[0])
    ds.FrameOfReferenceUID = "1.2.3.999"
    ds.save_as(paths[0])
    assert _codes(paths) == "INCONSISTENT_FRAME_OF_REFERENCE"


def test_empty_and_too_few(tmp_path: Path) -> None:
    assert _codes([]) == "EMPTY_INPUT"
    paths = make_ct_series(tmp_path)
    assert _codes(paths[:1]) == "TOO_FEW_SLICES"


def test_error_messages_contain_no_phi(tmp_path: Path) -> None:
    paths = make_ct_series(tmp_path)
    ds = pydicom.dcmread(paths[0])
    ds.Modality = "MR"
    ds.save_as(paths[0])
    with pytest.raises(IngestionError) as info:
        load_series(paths)
    assert "SYNTHETIC" not in info.value.message
    assert "SYNTH0001" not in info.value.message
