"""Postprocessing and quality checks in the reference (source series) geometry.

Voxel coordinates (``*_voxel``, ordered ``k, j, i``) and physical coordinates
(``*_mm``, DICOM patient LPS) are always reported in separate fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from scipy import ndimage

from ctpipeline.adapters.base import SegmentationPrediction, SegmentClass
from ctpipeline.common.errors import QualityCheckError
from ctpipeline.common.hashing import sha256_json
from ctpipeline.ingestion.geometry import VolumeGeometry
from ctpipeline.preprocessing.pipeline import PreprocessedVolume, labels_to_reference


class PostprocessingConfig(BaseModel):
    version: str = "1.0.0"
    min_component_voxels: int = Field(default=0, ge=0)
    connectivity: int = Field(default=1, ge=1, le=3)
    fail_on_empty: bool = False

    @property
    def sha256(self) -> str:
        return sha256_json(self.model_dump(mode="json"))


@dataclass
class Finding:
    finding_id: str
    label: int
    class_name: str
    voxel_count: int
    volume_mm3: float
    centroid_voxel: tuple[float, float, float]
    centroid_mm: tuple[float, float, float]
    bbox_voxel: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "label": self.label,
            "class_name": self.class_name,
            "voxel_count": self.voxel_count,
            "volume_mm3": self.volume_mm3,
            "centroid_voxel_kji": list(self.centroid_voxel),
            "centroid_mm_lps": list(self.centroid_mm),
            "bbox_voxel_kji": [list(b) for b in self.bbox_voxel],
        }


@dataclass
class PostprocessedResult:
    label_map: np.ndarray  # reference grid (k, j, i)
    classes: list[SegmentClass]
    findings: list[Finding]
    geometry: VolumeGeometry
    synthetic: bool
    qc_warnings: list[str] = field(default_factory=list)
    config_sha256: str = ""


def _remove_small_components(
    mask: np.ndarray, min_voxels: int, structure: np.ndarray
) -> np.ndarray:
    if min_voxels <= 1:
        return mask
    components, count = ndimage.label(mask, structure=structure)
    if count == 0:
        return mask
    sizes = np.bincount(components.ravel())
    keep = sizes >= min_voxels
    keep[0] = False
    return keep[components]


def postprocess(
    prediction: SegmentationPrediction,
    pre: PreprocessedVolume,
    config: PostprocessingConfig,
) -> PostprocessedResult:
    reference = pre.reference_geometry
    labels = labels_to_reference(prediction.label_map.astype(np.uint16), pre)

    qc: list[str] = []
    if tuple(labels.shape) != tuple(reference.shape):
        raise QualityCheckError("Mapped label map does not match the reference geometry")
    known = {c.label for c in prediction.classes} | {0}
    unknown = set(np.unique(labels).tolist()) - known
    if unknown:
        raise QualityCheckError(
            f"Model produced undeclared labels: {sorted(unknown)}", code="QC_UNKNOWN_LABELS"
        )

    structure = ndimage.generate_binary_structure(3, config.connectivity)
    cleaned = np.zeros_like(labels)
    findings: list[Finding] = []
    voxel_volume = float(np.prod(reference.spacing))
    for cls in prediction.classes:
        mask = _remove_small_components(labels == cls.label, config.min_component_voxels, structure)
        cleaned[mask] = cls.label
        components, count = ndimage.label(mask, structure=structure)
        for idx, slc in enumerate(ndimage.find_objects(components), start=1):
            if slc is None:
                continue
            comp = components == idx
            voxels = int(comp.sum())
            centroid = tuple(float(v) for v in ndimage.center_of_mass(comp))
            centroid_mm = tuple(float(v) for v in reference.index_to_physical(centroid))
            findings.append(
                Finding(
                    finding_id=f"L{cls.label}-C{idx}",
                    label=cls.label,
                    class_name=cls.name,
                    voxel_count=voxels,
                    volume_mm3=voxels * voxel_volume,
                    centroid_voxel=centroid,  # type: ignore[arg-type]
                    centroid_mm=centroid_mm,  # type: ignore[arg-type]
                    bbox_voxel=tuple((s.start, s.stop) for s in slc),  # type: ignore[arg-type]
                )
            )
        if count == 0:
            qc.append(f"Class '{cls.name}' is empty")

    if not cleaned.any():
        if config.fail_on_empty:
            raise QualityCheckError("Segmentation is empty", code="QC_EMPTY_OUTPUT")
        qc.append("Segmentation is empty")

    return PostprocessedResult(
        label_map=cleaned,
        classes=prediction.classes,
        findings=findings,
        geometry=reference,
        synthetic=prediction.synthetic,
        qc_warnings=qc,
        config_sha256=config.sha256,
    )
