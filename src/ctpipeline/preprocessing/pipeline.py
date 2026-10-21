"""Geometry-aware preprocessing.

Every operation is driven by a versioned :class:`PreprocessingConfig` whose
SHA-256 is recorded in the job manifest. Each transform that changes the voxel
grid is recorded so model outputs can be mapped back to the reference series.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, Field, model_validator
from scipy import ndimage

from ctpipeline.common.errors import PreprocessingError
from ctpipeline.common.hashing import sha256_json
from ctpipeline.ingestion.geometry import VolumeGeometry


class WindowConfig(BaseModel):
    center: float
    width: float = Field(gt=0)


class ResampleConfig(BaseModel):
    target_spacing: tuple[float, float, float]  # (i, j, k) mm
    order: int = Field(default=1, ge=0, le=3)

    @model_validator(mode="after")
    def _positive(self) -> ResampleConfig:
        if any(v <= 0 for v in self.target_spacing):
            raise ValueError("target_spacing must be positive")
        return self


class PreprocessingConfig(BaseModel):
    version: str
    window: WindowConfig | None = None
    normalization: Literal["none", "window_0_1", "zscore"] = "window_0_1"
    resample: ResampleConfig | None = None
    dtype: Literal["float32"] = "float32"

    @property
    def sha256(self) -> str:
        return sha256_json(self.model_dump(mode="json"))

    @classmethod
    def from_yaml(cls, path: str | Path) -> PreprocessingConfig:
        with open(path, encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))


@dataclass
class PreprocessedVolume:
    """Model-ready volume plus everything needed to undo the preprocessing."""

    volume: np.ndarray  # (k, j, i)
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: tuple[float, ...]
    metadata: dict[str, Any]
    config_version: str
    config_sha256: str
    reference_geometry: VolumeGeometry
    transforms: list[dict[str, Any]] = field(default_factory=list)

    @property
    def geometry(self) -> VolumeGeometry:
        return VolumeGeometry(
            origin=self.origin,
            spacing=self.spacing,
            direction=self.direction,
            shape=tuple(int(v) for v in self.volume.shape),  # type: ignore[arg-type]
        )


def _resample(
    volume: np.ndarray, geometry: VolumeGeometry, cfg: ResampleConfig
) -> tuple[np.ndarray, VolumeGeometry, dict[str, Any]]:
    """Resample keeping the physical extent of the volume fixed.

    Uses the "grid mode" convention (voxels are cells whose edges are aligned
    before and after), so the new origin is shifted by half the spacing change.
    """
    in_shape = np.asarray(volume.shape)
    in_spacing_kji = np.asarray(geometry.spacing[::-1])
    target_kji = np.asarray(cfg.target_spacing[::-1])
    out_shape = np.maximum(1, np.round(in_shape * in_spacing_kji / target_kji)).astype(int)
    out_spacing_kji = in_spacing_kji * in_shape / out_shape
    zoom = out_shape / in_shape
    out = ndimage.zoom(volume, zoom, order=cfg.order, mode="nearest", grid_mode=True)
    if tuple(out.shape) != tuple(out_shape):  # pragma: no cover - defensive
        raise PreprocessingError("Resampling produced an unexpected shape")
    shift_kji = (out_spacing_kji - in_spacing_kji) / 2.0
    shift_ijk = shift_kji[::-1]
    origin = np.asarray(geometry.origin) + geometry.direction_matrix @ shift_ijk
    new_geometry = VolumeGeometry(
        origin=tuple(float(v) for v in origin),  # type: ignore[arg-type]
        spacing=tuple(float(v) for v in out_spacing_kji[::-1]),  # type: ignore[arg-type]
        direction=geometry.direction,
        shape=tuple(int(v) for v in out_shape),  # type: ignore[arg-type]
    )
    record = {
        "op": "resample",
        "order": cfg.order,
        "input_shape": [int(v) for v in in_shape],
        "output_shape": [int(v) for v in out_shape],
        "input_spacing": list(geometry.spacing),
        "output_spacing": list(new_geometry.spacing),
    }
    return out.astype(np.float32), new_geometry, record


def preprocess(
    volume_hu: np.ndarray, geometry: VolumeGeometry, config: PreprocessingConfig
) -> PreprocessedVolume:
    if volume_hu.shape != geometry.shape:
        raise PreprocessingError("Volume shape does not match its geometry")
    if not np.isfinite(volume_hu).all():
        raise PreprocessingError("Volume contains non-finite values")

    transforms: list[dict[str, Any]] = []
    vol = volume_hu.astype(np.float32, copy=True)
    work_geometry = geometry

    if config.resample is not None:
        vol, work_geometry, record = _resample(vol, geometry, config.resample)
        transforms.append(record)

    if config.window is not None:
        lo = config.window.center - config.window.width / 2.0
        hi = config.window.center + config.window.width / 2.0
        vol = np.clip(vol, lo, hi)
        transforms.append({"op": "window", "low": lo, "high": hi})
        if config.normalization == "window_0_1":
            vol = (vol - lo) / (hi - lo)
            transforms.append({"op": "normalize", "method": "window_0_1"})
    elif config.normalization == "window_0_1":
        raise PreprocessingError("window_0_1 normalization requires a window")

    if config.normalization == "zscore":
        std = float(vol.std())
        mean = float(vol.mean())
        vol = (vol - mean) / (std if std > 0 else 1.0)
        transforms.append({"op": "normalize", "method": "zscore", "mean": mean, "std": std})

    return PreprocessedVolume(
        volume=vol.astype(np.float32),
        spacing=work_geometry.spacing,
        origin=work_geometry.origin,
        direction=work_geometry.direction,
        metadata={"input_geometry": geometry.to_dict(), "hu_range": _range(volume_hu)},
        config_version=config.version,
        config_sha256=config.sha256,
        reference_geometry=geometry,
        transforms=transforms,
    )


def _range(arr: np.ndarray) -> list[float]:
    return [float(arr.min()), float(arr.max())]


def labels_to_reference(labels: np.ndarray, pre: PreprocessedVolume) -> np.ndarray:
    """Map a label map from model space back to the reference series grid.

    Nearest-neighbour sampling of voxel centres (grid-mode convention matching
    :func:`_resample`), so label values are never interpolated.
    """
    if labels.shape != pre.volume.shape:
        raise PreprocessingError("Label map shape does not match the preprocessed volume")
    ref_shape = pre.reference_geometry.shape
    if tuple(labels.shape) == tuple(ref_shape):
        return labels.copy()
    indices = []
    for n_src, n_ref in zip(labels.shape, ref_shape, strict=True):
        centres = (np.arange(n_ref) + 0.5) * n_src / n_ref
        indices.append(np.clip(np.floor(centres).astype(int), 0, n_src - 1))
    return labels[np.ix_(*indices)]
