"""Synthetic CT series generator for tests and demos.

Generated data contains no real patient information.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from ctpipeline.ingestion.loader import CT_IMAGE_STORAGE


def make_phantom(shape: tuple[int, int, int] = (12, 32, 32), seed: int = 0) -> np.ndarray:
    """Return an HU phantom ``(k, j, i)``: air background, water body, dense sphere."""
    rng = np.random.default_rng(seed)
    k, j, i = np.indices(shape)
    centre = np.asarray(shape) / 2.0
    vol = np.full(shape, -1000.0)
    body = ((j - centre[1]) ** 2 + (i - centre[2]) ** 2) <= (min(shape[1:]) * 0.45) ** 2
    vol[body] = 0.0
    radius = min(shape) * 0.3
    sphere = ((k - centre[0]) ** 2 + (j - centre[1]) ** 2 + (i - centre[2]) ** 2) <= radius**2
    vol[sphere] = 300.0
    vol += rng.normal(0, 5, size=shape)
    return vol


def make_ct_series(
    out_dir: str | Path,
    volume_hu: np.ndarray | None = None,
    *,
    spacing: tuple[float, float, float] = (0.8, 0.8, 2.5),
    origin: tuple[float, float, float] = (-100.0, -120.0, 50.0),
    row_cosine: Sequence[float] = (1.0, 0.0, 0.0),
    col_cosine: Sequence[float] = (0.0, 1.0, 0.0),
    slope: float = 1.0,
    intercept: float = -1024.0,
    shuffle_names: bool = True,
    study_uid: str | None = None,
    series_uid: str | None = None,
    frame_of_reference_uid: str | None = None,
    seed: int = 0,
) -> list[Path]:
    """Write a single-frame CT series and return the written paths.

    File names are deliberately shuffled so tests prove that ordering comes
    from geometry, not names.
    """
    volume = make_phantom(seed=seed) if volume_hu is None else volume_hu
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    row = np.asarray(row_cosine, dtype=float)
    col = np.asarray(col_cosine, dtype=float)
    normal = np.cross(row, col)
    study_uid = study_uid or generate_uid()
    series_uid = series_uid or generate_uid()
    for_uid = frame_of_reference_uid or generate_uid()
    n = volume.shape[0]
    order = np.random.default_rng(seed).permutation(n) if shuffle_names else np.arange(n)
    paths = []
    for k in range(n):
        stored = np.round((volume[k] - intercept) / slope).astype(np.uint16)
        ipp = np.asarray(origin) + normal * spacing[2] * k
        sop_uid = generate_uid()
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CT_IMAGE_STORAGE
        meta.MediaStorageSOPInstanceUID = sop_uid
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID = CT_IMAGE_STORAGE
        ds.SOPInstanceUID = sop_uid
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = for_uid
        ds.Modality = "CT"
        ds.PatientName = "SYNTHETIC^PHANTOM"
        ds.PatientID = "SYNTH0001"
        ds.PatientBirthDate = ""
        ds.PatientSex = "O"
        ds.StudyDate = "20260101"
        ds.StudyTime = "120000"
        ds.StudyID = "1"
        ds.AccessionNumber = ""
        ds.ReferringPhysicianName = ""
        ds.Manufacturer = "SYNTHETIC"
        ds.SeriesNumber = 1
        ds.InstanceNumber = int(order[k]) + 1
        ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
        ds.ImagePositionPatient = [f"{v:.6f}" for v in ipp]
        ds.ImageOrientationPatient = [f"{v:.6f}" for v in (*row, *col)]
        ds.PixelSpacing = [f"{spacing[1]:.6f}", f"{spacing[0]:.6f}"]
        ds.SliceThickness = f"{spacing[2]:.6f}"
        ds.PositionReferenceIndicator = ""
        ds.KVP = "120"
        ds.AcquisitionNumber = 1
        ds.Rows, ds.Columns = stored.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.RescaleSlope = f"{slope:g}"
        ds.RescaleIntercept = f"{intercept:g}"
        ds.RescaleType = "HU"
        ds.PixelData = stored.tobytes()
        path = out / f"img_{int(order[k]):04d}.dcm"
        ds.save_as(path, enforce_file_format=True)
        paths.append(path)
    return paths


def main() -> None:
    """``python -m ctpipeline.testing.synthetic OUT_DIR`` writes a demo series."""
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "./data/synthetic-series"
    paths = make_ct_series(out)
    print(f"Wrote {len(paths)} synthetic CT slices to {out}")


if __name__ == "__main__":
    main()
