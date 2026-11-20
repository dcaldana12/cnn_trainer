"""In-memory dataset representation shared by every block."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .utils import strip_split_prefix

KNOWN_SPLITS = ("train", "val", "test")
_KNOWN_SPLIT_SET = frozenset(KNOWN_SPLITS)


@dataclass
class Annotation:
    """An object with pixel bounds and optional polygon vertices."""

    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    polygon: tuple[tuple[float, float], ...] | None = None

    def extends_outside(self, width: int, height: int) -> bool:
        # YOLO center and size are rounded separately, making edge errors possible.
        tolerance = 1e-6 * max(width, height, 1)
        return (
            self.x1 < -tolerance
            or self.y1 < -tolerance
            or self.x2 > width + tolerance
            or self.y2 > height + tolerance
        )

    def to_yolo(self, width: int, height: int) -> tuple[float, float, float, float]:
        """Convert to normalized ``(x_center, y_center, width, height)``."""
        return (
            (self.x1 + self.x2) / 2.0 / width,
            (self.y1 + self.y2) / 2.0 / height,
            (self.x2 - self.x1) / width,
            (self.y2 - self.y1) / height,
        )


@dataclass
class ImageRecord:
    """One image and its annotations."""

    path: Path
    relative_path: Path
    width: int
    height: int
    annotations: list[Annotation] = field(default_factory=list)
    split: str | None = None
    has_label: bool = True


@dataclass
class Dataset:
    """A format-independent collection of images, boxes and class names."""

    images: list[ImageRecord]
    classes: list[str]
    source_format: str
    root: Path
    orphans: list[Path] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.images)

    def class_index(self, name: str) -> int:
        try:
            return self.classes.index(name)
        except ValueError as error:
            raise KeyError(f"Unknown class name: {name!r}") from error

    def label_collisions(self) -> list[str]:
        """Return ``split:stem`` keys that would collide on YOLO export.

        YOLO pairs an image and its label by stem, so ``a.jpg`` and ``a.png``
        in the same split and directory cannot coexist. Images with the same
        name in different splits are fine.
        """
        stems = Counter(
            (
                record.split or "unassigned",
                strip_split_prefix(record.relative_path, _KNOWN_SPLIT_SET)
                .with_suffix("")
                .as_posix(),
            )
            for record in self.images
        )
        return sorted(f"{split}:{stem}" for (split, stem), count in stems.items() if count > 1)

    def stats(self) -> dict:
        class_counts = {name: 0 for name in self.classes}
        split_counts: dict[str, int] = {name: 0 for name in KNOWN_SPLITS}
        unassigned = 0
        empty_images = 0
        missing_labels = 0
        annotations = 0

        for record in self.images:
            if record.split in split_counts:
                split_counts[record.split] += 1
            else:
                unassigned += 1
            if not record.annotations:
                empty_images += 1
            if not record.has_label:
                missing_labels += 1
            for annotation in record.annotations:
                annotations += 1
                if 0 <= annotation.class_id < len(self.classes):
                    class_counts[self.classes[annotation.class_id]] += 1

        if unassigned:
            split_counts["unassigned"] = unassigned

        return {
            "images": len(self.images),
            "annotations": annotations,
            "empty_images": empty_images,
            "missing_labels": missing_labels,
            "orphan_labels": len(self.orphans),
            "classes": class_counts,
            "splits": split_counts,
        }
