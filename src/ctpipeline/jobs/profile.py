from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from ctpipeline.common.errors import ModelError
from ctpipeline.common.hashing import sha256_json
from ctpipeline.postprocessing.pipeline import PostprocessingConfig
from ctpipeline.preprocessing.pipeline import PreprocessingConfig


class ModelSection(BaseModel):
    name: str
    artifact: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class OverlaySection(BaseModel):
    window_center: float = 40.0
    window_width: float = 400.0
    alpha: float = Field(default=0.45, ge=0, le=1)


class PipelineProfile(BaseModel):
    """Versioned configuration of one model's pipeline (``configs/models/<name>.yaml``)."""

    model: ModelSection
    preprocessing: PreprocessingConfig
    postprocessing: PostprocessingConfig = Field(default_factory=PostprocessingConfig)
    overlay: OverlaySection = Field(default_factory=OverlaySection)

    @property
    def sha256(self) -> str:
        return sha256_json(self.model_dump(mode="json"))


def load_profile(configs_dir: str | Path, model_name: str) -> PipelineProfile:
    if not model_name.replace("-", "").replace("_", "").isalnum():
        raise ModelError("Invalid model name", code="UNKNOWN_MODEL")
    path = Path(configs_dir) / "models" / f"{model_name}.yaml"
    if not path.is_file():
        raise ModelError(f"No pipeline profile for model '{model_name}'", code="UNKNOWN_MODEL")
    with open(path, encoding="utf-8") as fh:
        return PipelineProfile.model_validate(yaml.safe_load(fh))
