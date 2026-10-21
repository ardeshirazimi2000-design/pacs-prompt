from __future__ import annotations

from dataclasses import dataclass

from highdicom.sr import CodedConcept as HdCodedConcept

from ctpipeline import __version__
from ctpipeline.adapters.base import CodedConcept, ModelInfo

MANUFACTURER = "ctpipeline (research prototype)"
MANUFACTURER_MODEL_NAME = "ctpipeline"
DEVICE_SERIAL_NUMBER = "NONE"
SOFTWARE_VERSIONS = __version__


def to_hd(code: CodedConcept) -> HdCodedConcept:
    return HdCodedConcept(value=code.value, scheme_designator=code.scheme, meaning=code.meaning)


@dataclass(frozen=True)
class ExportContext:
    """Identifiers of the outputs of one job."""

    model: ModelInfo
    series_number_base: int = 9000

    @property
    def algorithm_name(self) -> str:
        prefix = "SYNTHETIC " if self.model.synthetic else ""
        return (prefix + self.model.name)[:64]
