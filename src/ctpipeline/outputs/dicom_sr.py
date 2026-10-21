"""DICOM Structured Report export: TID 1500 "Measurement Report".

One TID 1411 "Volumetric ROI Measurements" group is written per exported
segment. Each group references its segment in the DICOM SEG object and holds
the segment volume (SCT 118565006 "Volume", UCUM ``mm3``). Component-level
findings are kept in the JSON results; the SR stays segment-level so every
measurement has an unambiguous DICOM reference.
"""

from __future__ import annotations

from collections.abc import Sequence

import highdicom as hd
import numpy as np
from pydicom.dataset import Dataset
from pydicom.sr.codedict import codes
from pydicom.uid import generate_uid

from ctpipeline.common.errors import ExportError
from ctpipeline.outputs.common import MANUFACTURER, ExportContext, to_hd
from ctpipeline.postprocessing.pipeline import PostprocessedResult

UCUM_MM3 = hd.sr.CodedConcept(value="mm3", scheme_designator="UCUM", meaning="cubic millimeter")


def build_measurement_report(
    source_datasets: Sequence[Dataset],
    segmentation: hd.seg.Segmentation,
    result: PostprocessedResult,
    ctx: ExportContext,
) -> hd.sr.Comprehensive3DSR:
    if not result.classes:
        raise ExportError("No segments to report", code="SR_NO_SEGMENTS")

    algorithm = hd.sr.AlgorithmIdentification(name=ctx.algorithm_name, version=ctx.model.version)
    observer = hd.sr.ObserverContext(
        observer_type=codes.DCM.Device,
        observer_identifying_attributes=hd.sr.DeviceObserverIdentifyingAttributes(
            uid=generate_uid(),
            name=ctx.algorithm_name,
            manufacturer_name=MANUFACTURER,
        ),
    )
    voxel_volume = float(np.prod(result.geometry.spacing))
    source_series = hd.sr.SourceSeriesForSegmentation(str(source_datasets[0].SeriesInstanceUID))

    groups = []
    for number, cls in enumerate(result.classes, start=1):
        volume_mm3 = float((result.label_map == cls.label).sum()) * voxel_volume
        groups.append(
            hd.sr.VolumetricROIMeasurementsAndQualitativeEvaluations(
                tracking_identifier=hd.sr.TrackingIdentifier(
                    uid=generate_uid(), identifier=f"segment-{number}"
                ),
                referenced_segment=hd.sr.ReferencedSegment(
                    sop_class_uid=segmentation.SOPClassUID,
                    sop_instance_uid=segmentation.SOPInstanceUID,
                    segment_number=number,
                    source_series=source_series,
                ),
                finding_type=to_hd(cls.property_type),
                algorithm_id=algorithm,
                measurements=[
                    hd.sr.Measurement(
                        name=codes.SCT.Volume,
                        value=volume_mm3,
                        unit=UCUM_MM3,
                        algorithm_id=algorithm,
                    )
                ],
            )
        )

    report = hd.sr.MeasurementReport(
        observation_context=hd.sr.ObservationContext(observer_device_context=observer),
        procedure_reported=codes.LN.CTUnspecifiedBodyRegion,
        imaging_measurements=groups,
    )
    return hd.sr.Comprehensive3DSR(
        evidence=[*source_datasets, segmentation],
        content=report,
        series_instance_uid=generate_uid(),
        series_number=ctx.series_number_base + 2,
        sop_instance_uid=generate_uid(),
        instance_number=1,
        manufacturer=MANUFACTURER,
        is_complete=True,
        is_final=False,
        is_verified=False,
    )
