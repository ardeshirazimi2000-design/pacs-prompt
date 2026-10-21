"""PNG overlay for human review. NOT a replacement for DICOM SEG/SR.

The colour of each segment is fixed by its position in the class list and
documented in ``SEGMENT_COLOURS``. File names never contain patient data.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw

from ctpipeline.postprocessing.pipeline import PostprocessedResult

SEGMENT_COLOURS: list[tuple[int, int, int]] = [
    (230, 25, 75),
    (60, 180, 75),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
]


def choose_slice(label_map: np.ndarray) -> int:
    """Slice with the largest labelled area, or the middle slice if empty."""
    areas = (label_map > 0).reshape(label_map.shape[0], -1).sum(axis=1)
    return int(areas.argmax()) if areas.any() else label_map.shape[0] // 2


def render_overlay(
    volume_hu: np.ndarray,
    result: PostprocessedResult,
    *,
    slice_index: int | None = None,
    window_center: float = 40.0,
    window_width: float = 400.0,
    alpha: float = 0.45,
) -> tuple[bytes, dict[str, object]]:
    """Return PNG bytes and metadata describing what was rendered."""
    if volume_hu.shape != result.label_map.shape:
        raise ValueError("CT volume and label map are not on the same grid")
    k = choose_slice(result.label_map) if slice_index is None else slice_index
    lo, hi = window_center - window_width / 2, window_center + window_width / 2
    grey = ((np.clip(volume_hu[k], lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)
    rgb = np.repeat(grey[..., None], 3, axis=-1).astype(np.float32)
    legend = []
    for idx, cls in enumerate(result.classes):
        colour = SEGMENT_COLOURS[idx % len(SEGMENT_COLOURS)]
        mask = result.label_map[k] == cls.label
        rgb[mask] = (1 - alpha) * rgb[mask] + alpha * np.asarray(colour, dtype=np.float32)
        legend.append({"label": cls.label, "name": cls.name, "rgb": list(colour)})

    image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    banner_h = 14
    canvas = Image.new("RGB", (image.width, image.height + banner_h), (0, 0, 0))
    canvas.paste(image, (0, banner_h))
    text = "SYNTHETIC - NOT FOR CLINICAL USE" if result.synthetic else "REVIEW ONLY"
    ImageDraw.Draw(canvas).text((2, 1), text, fill=(255, 255, 0))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    meta: dict[str, object] = {
        "slice_index": k,
        "window": [window_center, window_width],
        "alpha": alpha,
        "legend": legend,
        "synthetic": result.synthetic,
    }
    return buf.getvalue(), meta
