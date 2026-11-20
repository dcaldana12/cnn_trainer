"""Convert polygon annotations to detection boxes."""

from __future__ import annotations

from ..context import Context
from ..dataset import Dataset
from ..registry import block


@block("segmentation_to_boxes", "Convert YOLO polygons to axis-aligned detection boxes.")
def segmentation_to_boxes(dataset: Dataset, ctx: Context) -> Dataset:
    """Keep the tight enclosing bounds and discard the polygon vertices."""
    converted = 0
    for record in dataset.images:
        for annotation in record.annotations:
            if annotation.polygon is not None:
                annotation.polygon = None
                converted += 1
    ctx.logger.info("Converted %d polygon annotations to axis-aligned boxes", converted)
    return dataset
