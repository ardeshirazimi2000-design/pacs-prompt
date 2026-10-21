"""Patient-space geometry of a reconstructed volume.

Conventions (documented in docs/geometry.md):

* Arrays are indexed ``[k, j, i]`` = ``[slice, row, column]``.
* ``spacing`` is ``(column_spacing, row_spacing, slice_spacing)`` in mm, i.e.
  ordered like ``(i, j, k)``, which matches SimpleITK/ITK.
* ``direction`` is a 3x3 matrix whose *columns* are the unit vectors of the
  ``i`` (along a row), ``j`` (down a column) and ``k`` (slice stacking) axes in
  the DICOM patient coordinate system (LPS).
* ``origin`` is the patient position (mm) of voxel ``[0, 0, 0]`` centre, i.e.
  the ImagePositionPatient of the first slice after geometric sorting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VolumeGeometry:
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    direction: tuple[float, ...]  # 9 values, row-major 3x3, columns = i, j, k axes
    shape: tuple[int, int, int]  # (k, j, i)

    @property
    def direction_matrix(self) -> np.ndarray:
        return np.asarray(self.direction, dtype=float).reshape(3, 3)

    @property
    def affine(self) -> np.ndarray:
        """4x4 matrix mapping homogeneous ``(i, j, k, 1)`` to LPS mm."""
        mat = np.eye(4)
        mat[:3, :3] = self.direction_matrix * np.asarray(self.spacing)[None, :]
        mat[:3, 3] = self.origin
        return mat

    def index_to_physical(self, kji: np.ndarray | tuple[float, float, float]) -> np.ndarray:
        """Map ``(k, j, i)`` voxel coordinates (array of shape (..., 3)) to LPS mm."""
        kji_arr = np.asarray(kji, dtype=float)
        ijk = kji_arr[..., ::-1]
        homog = np.concatenate([ijk, np.ones(ijk.shape[:-1] + (1,))], axis=-1)
        return (homog @ self.affine.T)[..., :3]

    def physical_to_index(self, xyz: np.ndarray | tuple[float, float, float]) -> np.ndarray:
        """Inverse of :meth:`index_to_physical`; returns ``(k, j, i)``."""
        xyz_arr = np.asarray(xyz, dtype=float)
        homog = np.concatenate([xyz_arr, np.ones(xyz_arr.shape[:-1] + (1,))], axis=-1)
        ijk = (homog @ np.linalg.inv(self.affine).T)[..., :3]
        return ijk[..., ::-1]

    def to_dict(self) -> dict[str, list[float] | list[int]]:
        return {
            "origin": list(self.origin),
            "spacing": list(self.spacing),
            "direction": list(self.direction),
            "shape": list(self.shape),
        }
