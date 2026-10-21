"""DICOM CT series ingestion and validation.

Scope of the MVP (see docs/dicom-support.md):

* Single-frame "CT Image Storage" objects only. Enhanced CT and other
  multi-frame objects are rejected explicitly.
* Uncompressed transfer syntaxes plus RLE Lossless (decoded natively by
  pydicom). Other encodings are rejected with ``UNSUPPORTED_TRANSFER_SYNTAX``.
* One series per job. When the input contains several series the caller must
  select one explicitly.
* Slices are ordered by projecting ImagePositionPatient onto the slice normal
  (row cosine x column cosine); file names are never used for ordering.
* Gantry-tilted (sheared) and non-uniformly spaced stacks are rejected.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError
from pydicom.uid import (
    UID,
    DeflatedExplicitVRLittleEndian,
    ExplicitVRBigEndian,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    RLELossless,
)

from ctpipeline.common.errors import IngestionError
from ctpipeline.common.hashing import sha256_bytes, sha256_manifest
from ctpipeline.ingestion.geometry import VolumeGeometry

CT_IMAGE_STORAGE = UID("1.2.840.10008.5.1.4.1.1.2")
ENHANCED_CT_IMAGE_STORAGE = UID("1.2.840.10008.5.1.4.1.1.2.1")

SUPPORTED_TRANSFER_SYNTAXES = frozenset(
    {
        ImplicitVRLittleEndian,
        ExplicitVRLittleEndian,
        ExplicitVRBigEndian,
        DeflatedExplicitVRLittleEndian,
        RLELossless,
    }
)

REQUIRED_ATTRIBUTES = (
    "SOPClassUID",
    "SOPInstanceUID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "FrameOfReferenceUID",
    "Modality",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "PixelSpacing",
    "Rows",
    "Columns",
)

ORIENTATION_TOLERANCE = 1e-4
SPACING_RELATIVE_TOLERANCE = 0.01
POSITION_TOLERANCE_MM = 1e-3


@dataclass
class IngestionOptions:
    series_instance_uid: str | None = None
    min_slices: int = 2
    load_pixels: bool = True


@dataclass
class IngestedSeries:
    """A validated, geometrically sorted CT series."""

    datasets: list[Dataset]
    geometry: VolumeGeometry
    study_instance_uid: str
    series_instance_uid: str
    frame_of_reference_uid: str
    input_manifest_sha256: str
    volume_hu: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def sop_instance_uids(self) -> list[str]:
        return [str(ds.SOPInstanceUID) for ds in self.datasets]


def _read(path: Path) -> tuple[Dataset, str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise IngestionError("An input file could not be read", code="UNREADABLE_FILE") from exc
    try:
        ds = pydicom.dcmread(path, force=False)
    except (InvalidDicomError, EOFError, ValueError, TypeError, AttributeError) as exc:
        raise IngestionError(
            f"Input file '{path.name}' is not a valid DICOM Part 10 file",
            code="INVALID_DICOM_FILE",
        ) from exc
    return ds, sha256_bytes(raw)


def _check_instance(ds: Dataset, name: str) -> None:
    missing = [attr for attr in REQUIRED_ATTRIBUTES if attr not in ds]
    if missing:
        raise IngestionError(
            f"File '{name}' is missing required attributes: {', '.join(missing)}",
            code="MISSING_ATTRIBUTES",
        )
    if ds.SOPClassUID == ENHANCED_CT_IMAGE_STORAGE or int(ds.get("NumberOfFrames", 1) or 1) > 1:
        raise IngestionError(
            "Enhanced/multi-frame CT is not supported by this version",
            code="UNSUPPORTED_MULTIFRAME",
        )
    if ds.SOPClassUID != CT_IMAGE_STORAGE:
        raise IngestionError(
            f"Unsupported SOP Class UID {ds.SOPClassUID}", code="UNSUPPORTED_SOP_CLASS"
        )
    if str(ds.Modality).upper() != "CT":
        raise IngestionError(f"Unsupported modality '{ds.Modality}'", code="UNSUPPORTED_MODALITY")
    ts = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
    if ts is None or ts not in SUPPORTED_TRANSFER_SYNTAXES:
        raise IngestionError(
            f"Unsupported transfer syntax {ts}", code="UNSUPPORTED_TRANSFER_SYNTAX"
        )
    if "PixelData" not in ds:
        raise IngestionError(f"File '{name}' has no pixel data", code="MISSING_PIXEL_DATA")


def _orientation(ds: Dataset) -> tuple[np.ndarray, np.ndarray]:
    iop = np.asarray([float(v) for v in ds.ImageOrientationPatient], dtype=float)
    row, col = iop[:3], iop[3:]
    if (
        abs(np.linalg.norm(row) - 1) > 1e-3
        or abs(np.linalg.norm(col) - 1) > 1e-3
        or abs(float(np.dot(row, col))) > 1e-3
    ):
        raise IngestionError(
            "ImageOrientationPatient is not orthonormal", code="INVALID_ORIENTATION"
        )
    return row, col


def _select_series(
    items: list[tuple[Dataset, str, str]], selected: str | None
) -> list[tuple[Dataset, str, str]]:
    by_series: dict[str, list[tuple[Dataset, str, str]]] = defaultdict(list)
    for item in items:
        by_series[str(item[0].SeriesInstanceUID)].append(item)
    if selected is not None:
        if selected not in by_series:
            raise IngestionError(
                "The requested SeriesInstanceUID is not present in the input",
                code="SERIES_NOT_FOUND",
            )
        return by_series[selected]
    if len(by_series) > 1:
        raise IngestionError(
            f"Input contains {len(by_series)} series; select one explicitly",
            code="MULTIPLE_SERIES",
        )
    return next(iter(by_series.values()))


def _deduplicate(
    items: list[tuple[Dataset, str, str]], warnings: list[str]
) -> list[tuple[Dataset, str, str]]:
    seen: dict[str, str] = {}
    unique = []
    for ds, digest, name in items:
        uid = str(ds.SOPInstanceUID)
        if uid in seen:
            if seen[uid] != digest:
                raise IngestionError(
                    "Two different files share the same SOPInstanceUID",
                    code="DUPLICATE_SOP_INSTANCE_CONFLICT",
                )
            warnings.append(f"Identical duplicate of an instance ignored ('{name}')")
            continue
        seen[uid] = digest
        unique.append((ds, digest, name))
    return unique


def load_series(
    paths: Sequence[str | Path], options: IngestionOptions | None = None
) -> IngestedSeries:
    """Read, validate and geometrically sort a CT series.

    Raises :class:`IngestionError` with a stable ``code`` for every rejected input.
    """
    opts = options or IngestionOptions()
    if not paths:
        raise IngestionError("No input files were provided", code="EMPTY_INPUT")

    items: list[tuple[Dataset, str, str]] = []
    for raw_path in paths:
        path = Path(raw_path)
        ds, digest = _read(path)
        _check_instance(ds, path.name)
        items.append((ds, digest, path.name))

    study_uids = {str(ds.StudyInstanceUID) for ds, _, _ in items}
    if len(study_uids) > 1:
        raise IngestionError("Input files belong to more than one study", code="MULTIPLE_STUDIES")

    warnings: list[str] = []
    items = _deduplicate(_select_series(items, opts.series_instance_uid), warnings)
    if len(items) < opts.min_slices:
        raise IngestionError(
            f"Series has {len(items)} slice(s); at least {opts.min_slices} required",
            code="TOO_FEW_SLICES",
        )

    first = items[0][0]
    for_uids = {str(ds.FrameOfReferenceUID) for ds, _, _ in items}
    if len(for_uids) != 1:
        raise IngestionError(
            "Slices do not share one FrameOfReferenceUID", code="INCONSISTENT_FRAME_OF_REFERENCE"
        )

    row, col = _orientation(first)
    for ds, _, _ in items[1:]:
        r, c = _orientation(ds)
        if np.abs(r - row).max() > ORIENTATION_TOLERANCE or np.abs(c - col).max() > (
            ORIENTATION_TOLERANCE
        ):
            raise IngestionError(
                "Slices have different ImageOrientationPatient", code="INCONSISTENT_ORIENTATION"
            )
        if (ds.Rows, ds.Columns) != (first.Rows, first.Columns):
            raise IngestionError("Slices have different matrix sizes", code="INCONSISTENT_SHAPE")
        if (
            np.abs(
                np.asarray(ds.PixelSpacing, dtype=float)
                - np.asarray(first.PixelSpacing, dtype=float)
            ).max()
            > 1e-4
        ):
            raise IngestionError(
                "Slices have different PixelSpacing", code="INCONSISTENT_PIXEL_SPACING"
            )

    normal = np.cross(row, col)
    positions = np.asarray(
        [[float(v) for v in ds.ImagePositionPatient] for ds, _, _ in items], dtype=float
    )
    projections = positions @ normal
    order = np.argsort(projections, kind="stable")
    items = [items[i] for i in order]
    positions = positions[order]
    projections = projections[order]

    gaps = np.diff(projections)
    if np.any(gaps < POSITION_TOLERANCE_MM):
        raise IngestionError(
            "Two or more slices share the same position", code="DUPLICATE_SLICE_POSITION"
        )
    slice_spacing = float(np.mean(gaps))
    if np.abs(gaps - slice_spacing).max() > SPACING_RELATIVE_TOLERANCE * slice_spacing:
        raise IngestionError(
            "Slice spacing is not uniform (missing or extra slices?)",
            code="NON_UNIFORM_SLICE_SPACING",
        )
    # Each slice must be displaced exactly along the normal (no gantry tilt / shear).
    offsets = positions - positions[0]
    in_plane = offsets - np.outer(offsets @ normal, normal)
    if np.abs(in_plane).max() > max(0.01, 0.01 * slice_spacing):
        raise IngestionError(
            "Slice positions are not aligned with the slice normal (gantry tilt)",
            code="UNSUPPORTED_GANTRY_TILT",
        )

    row_spacing, col_spacing = (float(v) for v in first.PixelSpacing)
    direction = np.stack([row, col, normal], axis=1)
    geometry = VolumeGeometry(
        origin=tuple(float(v) for v in positions[0]),  # type: ignore[arg-type]
        spacing=(col_spacing, row_spacing, slice_spacing),
        direction=tuple(float(v) for v in direction.reshape(-1)),
        shape=(len(items), int(first.Rows), int(first.Columns)),
    )

    series = IngestedSeries(
        datasets=[ds for ds, _, _ in items],
        geometry=geometry,
        study_instance_uid=str(first.StudyInstanceUID),
        series_instance_uid=str(first.SeriesInstanceUID),
        frame_of_reference_uid=for_uids.pop(),
        input_manifest_sha256=sha256_manifest(
            (str(ds.SOPInstanceUID), digest) for ds, digest, _ in items
        ),
        warnings=warnings,
    )
    if opts.load_pixels:
        series.volume_hu = _build_volume(series.datasets)
    return series


def _build_volume(datasets: Sequence[Dataset]) -> np.ndarray:
    slices = []
    for ds in datasets:
        try:
            pixels = ds.pixel_array.astype(np.float32)
        except Exception as exc:  # pydicom raises many types for undecodable data
            raise IngestionError(
                "Pixel data could not be decoded", code="PIXEL_DECODE_ERROR"
            ) from exc
        slope = float(ds.get("RescaleSlope", 1.0) or 1.0)
        intercept = float(ds.get("RescaleIntercept", 0.0) or 0.0)
        slices.append(pixels * np.float32(slope) + np.float32(intercept))
    return np.stack(slices, axis=0)
