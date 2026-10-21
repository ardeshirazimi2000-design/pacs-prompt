# Geometry conventions

* Volumes are NumPy arrays indexed `[k, j, i]` = `[slice, row, column]`.
* `spacing` = `(column spacing, row spacing, slice spacing)` mm, ordered `(i, j, k)`.
  DICOM `PixelSpacing` is `[row spacing, column spacing]`, so it is swapped on read.
* `direction` = 3x3 matrix (row-major) whose **columns** are the unit vectors of the
  `i`, `j`, `k` axes in DICOM patient coordinates (LPS):
  `i` = row cosine, `j` = column cosine (from `ImageOrientationPatient`),
  `k` = `row x column` (the slice normal).
* `origin` = `ImagePositionPatient` of slice `k = 0` after sorting.
* Slices are sorted by `dot(ImagePositionPatient, normal)` in ascending order. File
  names and `InstanceNumber` are never used.
* `VolumeGeometry.index_to_physical((k, j, i))` = `origin + D @ diag(spacing) @ (i, j, k)`.

## Rejected stacks

| Code | Condition |
|---|---|
| `NON_UNIFORM_SLICE_SPACING` | gaps between slices differ by more than 1 % (missing/extra slices) |
| `DUPLICATE_SLICE_POSITION` | two slices at the same position |
| `UNSUPPORTED_GANTRY_TILT` | slice offsets are not parallel to the normal |
| `INCONSISTENT_ORIENTATION` / `_SHAPE` / `_PIXEL_SPACING` / `_FRAME_OF_REFERENCE` | per-slice attributes disagree |

## Resampling

Resampling uses the *grid mode* convention (voxels are cells; the outer edges of the
volume stay fixed). Output shape = `round(n * old_spacing / target_spacing)`; the actual
output spacing is adjusted so that `n_out * spacing_out == n_in * spacing_in` exactly.
The origin moves by `D @ ((spacing_out - spacing_in) / 2)`.

Model label maps are mapped back to the reference grid with nearest-neighbour sampling of
voxel centres under the same convention (`labels_to_reference`), so labels are never
interpolated. Tests in `tests/unit/test_preprocessing.py` check that the physical
centroid of an object is unchanged by resampling.
