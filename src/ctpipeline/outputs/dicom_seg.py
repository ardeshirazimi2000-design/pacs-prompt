"""DICOM Segmentation (SEG) export via highdicom.

highdicom builds the Segmentation IOD (referenced series/instances, Frame of
Reference, Shared/Per-frame functional groups, Segment Sequence). The source
datasets passed in must be the geometrically sorted series from ingestion so
mask slice ``k`` is aligned with source instance ``k``.
"""

from __future__ import annotations

from collections.abc import Sequence

import highdicom as hd
import numpy as np
from pydicom.dataset import Dataset
from pydicom.sr.codedict import codes
from pydicom.uid import generate_uid

from ctpipeline.common.errors import ExportError
from ctpipeline.outputs.common import (
    DEVICE_SERIAL_NUMBER,
    MANUFACTURER,
    MANUFACTURER_MODEL_NAME,
    SOFTWARE_VERSIONS,
    ExportContext,
    to_hd,
)
from ctpipeline.postprocessing.pipeline import PostprocessedResult


def build_segmentation(
    source_datasets: Sequence[Dataset], result: PostprocessedResult, ctx: ExportContext
) -> hd.seg.Segmentation:
    if result.label_map.shape[0] != len(source_datasets):
        raise ExportError("Mask slice count differs from the source series", code="SEG_GEOMETRY")
    rows, cols = int(source_datasets[0].Rows), int(source_datasets[0].Columns)
    if result.label_map.shape[1:] != (rows, cols):
        raise ExportError("Mask matrix size differs from the source series", code="SEG_GEOMETRY")
    if not result.classes:
        raise ExportError("No segment classes to export", code="SEG_NO_SEGMENTS")

    algorithm = hd.AlgorithmIdentificationSequence(
        name=ctx.algorithm_name,
        family=codes.cid7162.ArtificialIntelligence,
        version=ctx.model.version,
        parameters={"weights_sha256": ctx.model.weights_sha256[:64]},
    )
    descriptions = []
    masks = []
    for number, cls in enumerate(result.classes, start=1):
        label = (
            cls.name
            if not result.synthetic or "SYNTHETIC" in cls.name.upper()
            else ("SYNTHETIC " + cls.name)
        )
        descriptions.append(
            hd.seg.SegmentDescription(
                segment_number=number,
                segment_label=label[:64],
                segmented_property_category=to_hd(cls.category),
                segmented_property_type=to_hd(cls.property_type),
                algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
                algorithm_identification=algorithm,
                tracking_uid=generate_uid(),
                tracking_id=f"segment-{number}",
            )
        )
        masks.append(result.label_map == cls.label)
    # (frames, rows, cols, segments) binary array; highdicom handles empty segments.
    pixel_array = np.stack(masks, axis=-1).astype(np.uint8)

    return hd.seg.Segmentation(
        source_images=list(source_datasets),
        pixel_array=pixel_array,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=descriptions,
        series_instance_uid=generate_uid(),
        series_number=ctx.series_number_base + 1,
        sop_instance_uid=generate_uid(),
        instance_number=1,
        manufacturer=MANUFACTURER,
        manufacturer_model_name=MANUFACTURER_MODEL_NAME,
        software_versions=SOFTWARE_VERSIONS,
        device_serial_number=DEVICE_SERIAL_NUMBER,
        content_description=(
            "SYNTHETIC - NOT FOR CLINICAL USE" if result.synthetic else "Research use only"
        ),
        content_label="CTPIPELINE",
        omit_empty_frames=False,
    )
