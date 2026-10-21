# DICOM support (version 0.1.0)

## Input

| Supported | Not supported (explicit error code) |
|---|---|
| CT Image Storage `1.2.840.10008.5.1.4.1.1.2`, single frame | Enhanced CT / multi-frame (`UNSUPPORTED_MULTIFRAME`) |
| Implicit/Explicit VR Little Endian, Explicit VR Big Endian, Deflated, RLE Lossless | JPEG, JPEG 2000, JPEG-LS and others (`UNSUPPORTED_TRANSFER_SYNTAX`) |
| One series per job (select with `series_instance_uid` when the upload has several) | Several series without selection (`MULTIPLE_SERIES`) |

Identical duplicate files are dropped with a warning; two different files with the same
`SOPInstanceUID` are rejected (`DUPLICATE_SOP_INSTANCE_CONFLICT`). `RescaleSlope` /
`RescaleIntercept` are applied per slice.

## Output

* **DICOM SEG** – built with highdicom 0.28.2 (`Segmentation`, BINARY, one segment per
  model class, `AUTOMATIC` algorithm type with `AlgorithmIdentificationSequence`, family
  DCM 123110 "Artificial Intelligence"). One frame per source slice
  (`omit_empty_frames=False`), referencing the source instances and Frame of Reference.
  Segment category/type codes come from the adapter; the dummy uses SNOMED CT 85756007
  "Tissue" and its segment labels start with `SYNTHETIC`.
* **DICOM SR** – Comprehensive 3D SR with TID 1500 Measurement Report. One TID 1411
  volumetric ROI group per segment, referencing the SEG segment, with the measurement
  SCT 118565006 "Volume" in UCUM `mm3`. Procedure reported: LOINC 25045-6
  "CT unspecified body region". The report is marked complete but **not verified**.
* **Overlay PNG** – review aid only, banner `SYNTHETIC - NOT FOR CLINICAL USE` for dummy
  outputs; never a replacement for SEG/SR.

## Validation status

* Unit tests re-read SEG/SR and check SOP classes, source references, Frame of Reference,
  per-frame `ImagePositionPatient` against the source slices, mask/frame alignment, empty
  and multi-segment masks, and SR → SEG references and values.
* The CI job `dicom-conformance` runs the independent validator
  [dicom-validator](https://github.com/pydicom/dicom-validator) on generated outputs. It
  is **informational** (`continue-on-error`) until its findings have been reviewed; it
  could not be run while developing because the build sandbox had no access to the DICOM
  standard download.
* Opening outputs in the target viewer (e.g. OHIF, 3D Slicer) has **not** been tested yet.

Do not consider SEG/SR conformant for any target system until both checks pass there.
