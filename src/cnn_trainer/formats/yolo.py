"""YOLO directory loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..dataset import KNOWN_SPLITS, Annotation, Dataset, ImageRecord
from ..errors import DatasetError
from ..utils import iter_image_files, read_image_size, resolve_path, safe_relative_path
from .base import normalize_class_names


def load(source: dict, root: Path, class_names: list[str] | None) -> Dataset:
    images_dir = _resolve_images_dir(source, root)
    labels_dir = _resolve_labels_dir(source, root, images_dir)
    data_path = _resolve_data_yaml(source, root)
    data_names, data_payload = _read_data_yaml(data_path)

    if class_names is None:
        class_names = data_names
    if class_names is None:
        declared_count = data_payload.get("nc")
        if (
            isinstance(declared_count, int)
            and not isinstance(declared_count, bool)
            and declared_count > 0
        ):
            class_names = [f"class_{index}" for index in range(declared_count)]
    names_explicit = class_names is not None

    image_paths = iter_image_files(images_dir)
    if not image_paths:
        raise DatasetError(f"No images found under {images_dir}")

    infer_splits = bool(source.get("infer_splits", True))
    records: list[ImageRecord] = []
    raw_annotations: dict[int, list[tuple[int, float, float, float, float]]] = {}
    parse_errors: list[str] = []
    max_class_id = -1

    for index, image_path in enumerate(image_paths):
        relative_path = safe_relative_path(image_path.relative_to(images_dir))
        label_path = labels_dir / relative_path.with_suffix(".txt")
        has_label = label_path.is_file()
        width, height = read_image_size(image_path)

        split = None
        if infer_splits and len(relative_path.parts) > 1:
            first = relative_path.parts[0]
            if first in KNOWN_SPLITS:
                split = first

        annotations: list[tuple[int, float, float, float, float]] = []
        if has_label:
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8", errors="replace").splitlines(),
                start=1,
            ):
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                if len(parts) != 5:
                    parse_errors.append(
                        f"{relative_path.with_suffix('.txt')}, line {line_number}: "
                        f"expected 5 values, found {len(parts)}"
                    )
                    continue
                try:
                    class_id = int(parts[0])
                    x_center, y_center, box_width, box_height = (
                        float(value) for value in parts[1:]
                    )
                except ValueError:
                    parse_errors.append(
                        f"{relative_path.with_suffix('.txt')}, line {line_number}: "
                        "non-numeric class or box value"
                    )
                    continue
                if class_id < 0:
                    parse_errors.append(
                        f"{relative_path.with_suffix('.txt')}, line {line_number}: "
                        f"negative class id {class_id}"
                    )
                    continue
                max_class_id = max(max_class_id, class_id)
                annotations.append((class_id, x_center, y_center, box_width, box_height))

        raw_annotations[index] = annotations
        records.append(
            ImageRecord(
                path=image_path.resolve(),
                relative_path=relative_path,
                width=width,
                height=height,
                split=split,
                has_label=has_label,
            )
        )

    if class_names is None:
        class_names = [f"class_{class_id}" for class_id in range(max_class_id + 1)]
    elif max_class_id >= len(class_names):
        if names_explicit:
            raise DatasetError(
                f"{labels_dir} contains class id {max_class_id}, but only "
                f"{len(class_names)} class name(s) are defined. "
                "Fix data.yaml or pass source.class_names."
            )
        class_names.extend(
            f"class_{class_id}" for class_id in range(len(class_names), max_class_id + 1)
        )

    for index, record in enumerate(records):
        for class_id, x_center, y_center, box_width, box_height in raw_annotations[index]:
            if class_id >= len(class_names):
                parse_errors.append(
                    f"{record.relative_path.with_suffix('.txt')}: class id {class_id} "
                    f"is outside the defined class list"
                )
                continue
            record.annotations.append(
                Annotation(
                    class_id=class_id,
                    x1=(x_center - box_width / 2.0) * record.width,
                    y1=(y_center - box_height / 2.0) * record.height,
                    x2=(x_center + box_width / 2.0) * record.width,
                    y2=(y_center + box_height / 2.0) * record.height,
                )
            )

    orphans = _find_orphan_labels(labels_dir, records)
    if not labels_dir.is_dir():
        parse_errors.append(f"Labels directory not found: {labels_dir}")

    return Dataset(
        images=records,
        classes=class_names,
        source_format="yolo",
        root=root,
        orphans=orphans,
        metadata={
            "images": str(images_dir),
            "labels": str(labels_dir),
            "data_yaml": str(data_path) if data_path else None,
            "data": data_payload,
            "parse_errors": parse_errors,
        },
    )


def _resolve_images_dir(source: dict, root: Path) -> Path:
    configured = source.get("images")
    if configured is not None:
        path = resolve_path(configured, root)
        if path is None or not path.is_dir():
            raise DatasetError(f"Images directory not found: {path}")
        return path

    default = root / "images"
    if default.is_dir():
        return default.resolve()
    return root


def _resolve_labels_dir(source: dict, root: Path, images_dir: Path) -> Path:
    configured = source.get("labels")
    if configured is not None:
        path = resolve_path(configured, root)
        if path is None:
            raise DatasetError("Invalid labels directory")
        return path

    # Mirror an ``images`` path component to ``labels``. This handles both
    # ``root/images`` and split layouts such as ``root/images/train`` without
    # assuming that the labels live directly under the dataset root.
    parts = list(images_dir.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        return Path(*parts).resolve()

    default = root / "labels"
    if default.is_dir():
        return default.resolve()
    return images_dir


def _resolve_data_yaml(source: dict, root: Path) -> Path | None:
    configured = source.get("data_yaml")
    if configured is not None:
        path = resolve_path(configured, root)
        if path is None or not path.is_file():
            raise DatasetError(f"YOLO data YAML not found: {path}")
        return path

    for name in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml"):
        candidate = root / name
        if candidate.is_file():
            return candidate.resolve()
    return None


def _read_data_yaml(path: Path | None) -> tuple[list[str] | None, dict[str, Any]]:
    if path is None:
        return None, {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as error:
        raise DatasetError(f"Could not read YOLO data YAML {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DatasetError(f"YOLO data YAML must contain a mapping: {path}")
    names = normalize_class_names(payload.get("names"))
    return names, payload


def _find_orphan_labels(labels_dir: Path, records: list[ImageRecord]) -> list[Path]:
    if not labels_dir.is_dir():
        return []

    expected = {
        (labels_dir / record.relative_path.with_suffix(".txt")).resolve() for record in records
    }
    orphans = [
        path.resolve()
        for path in labels_dir.rglob("*.txt")
        if path.is_file() and path.resolve() not in expected
    ]
    return sorted(orphans, key=lambda path: path.as_posix())
