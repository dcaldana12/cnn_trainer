"""Source dataset adapters."""

from __future__ import annotations

from pathlib import Path

from ..dataset import Dataset
from ..errors import ConfigError
from ..utils import resolve_path
from . import coco, cvat, yolo
from .base import normalize_class_names

__all__ = ["load_dataset", "normalize_class_names"]

_COMMON_SOURCE_KEYS = {"format", "root", "class_names"}
_FORMAT_KEYS = {
    "cvat_xml": {"annotations", "images", "skip_outside", "allow_unknown_labels"},
    "coco": {"annotations", "images", "skip_crowd"},
    "yolo": {"images", "labels", "data_yaml", "infer_splits"},
}
_ALIASES = {
    "cvat": "cvat_xml",
    "cvat-images": "cvat_xml",
    "cvat_images": "cvat_xml",
    "coco_json": "coco",
    "ultralytics": "yolo",
}


def load_dataset(source: dict, config_dir: Path) -> Dataset:
    """Load a dataset described by the ``source`` section of a pipeline."""
    if not isinstance(source, dict):
        raise ConfigError("The 'source' section must be a mapping")

    source_format = str(source.get("format", "")).strip().lower()
    if not source_format:
        raise ConfigError("The 'source' section needs a 'format' key")

    root_value = source.get("root", ".")
    root = resolve_path(root_value, config_dir)
    if root is None:
        raise ConfigError("Invalid source root")

    canonical = _ALIASES.get(source_format, source_format)
    if canonical not in _FORMAT_KEYS:
        raise ConfigError(
            f"Unknown source format: {source_format!r}. Supported formats: cvat_xml, coco, yolo"
        )

    allowed = _COMMON_SOURCE_KEYS | _FORMAT_KEYS[canonical]
    unknown = sorted(set(source) - allowed)
    if unknown:
        raise ConfigError(
            f"Unknown source option(s) for {canonical!r}: {', '.join(unknown)}. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )

    class_names = normalize_class_names(source.get("class_names"))

    if canonical == "cvat_xml":
        return cvat.load(source, root, class_names)
    if canonical == "coco":
        return coco.load(source, root, class_names)
    return yolo.load(source, root, class_names)
