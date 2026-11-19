"""COCO JSON loader."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ..dataset import Annotation, Dataset, ImageRecord
from ..errors import DatasetError
from ..utils import read_image_size, resolve_path, safe_relative_path


def load(source: dict, root: Path, class_names: list[str] | None) -> Dataset:
    annotations_path = _find_annotations(source, root)
    images_dir = resolve_path(source.get("images", "."), root)
    if images_dir is None or not images_dir.is_dir():
        images_dir = root

    try:
        payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetError(f"Could not read COCO JSON {annotations_path}: {error}") from error

    images = payload.get("images")
    annotations = payload.get("annotations")
    categories = payload.get("categories", [])
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise DatasetError(
            f"{annotations_path} is not a COCO detection file: "
            "expected 'images' and 'annotations' lists"
        )

    category_names: dict[int, str] = {}
    for category in categories:
        try:
            category_id = int(category["id"])
            name = str(category["name"])
        except (KeyError, TypeError, ValueError) as error:
            raise DatasetError(f"Invalid COCO category: {category!r}") from error
        if name in category_names.values():
            raise DatasetError(f"Duplicate COCO category name: {name!r}")
        category_names[category_id] = name

    if class_names is None:
        class_names = [category_names[cid] for cid in sorted(category_names)]
    else:
        class_names = list(class_names)
        missing = sorted(set(category_names.values()) - set(class_names))
        if missing:
            raise DatasetError(f"COCO categories missing from source.class_names: {missing}")

    if not class_names:
        raise DatasetError("No COCO categories found")

    category_to_index = {
        category_id: class_names.index(name) for category_id, name in category_names.items()
    }

    skip_crowd = bool(source.get("skip_crowd", True))
    annotations_by_image: dict[int, list[Annotation]] = defaultdict(list)
    parse_errors: list[str] = []

    for annotation in annotations:
        try:
            image_id = int(annotation["image_id"])
            category_id = int(annotation["category_id"])
            bbox = annotation["bbox"]
        except (KeyError, TypeError, ValueError):
            parse_errors.append(f"Invalid COCO annotation was skipped: {annotation!r}")
            continue

        if skip_crowd and annotation.get("iscrowd"):
            continue

        if category_id not in category_to_index:
            parse_errors.append(
                f"Annotation {annotation.get('id', '?')} uses unknown category "
                f"{category_id} and was skipped"
            )
            continue

        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            parse_errors.append(
                f"Annotation {annotation.get('id', '?')} has invalid bbox and was skipped"
            )
            continue

        try:
            x, y, width, height = (float(value) for value in bbox)
        except (TypeError, ValueError):
            parse_errors.append(
                f"Annotation {annotation.get('id', '?')} has invalid bbox and was skipped"
            )
            continue

        annotations_by_image[image_id].append(
            Annotation(
                class_id=category_to_index[category_id],
                x1=x,
                y1=y,
                x2=x + width,
                y2=y + height,
            )
        )

    records: list[ImageRecord] = []
    for image in images:
        try:
            image_id = int(image["id"])
            file_name = str(image["file_name"])
        except (KeyError, TypeError, ValueError):
            parse_errors.append(f"Invalid COCO image entry was skipped: {image!r}")
            continue

        relative_path = safe_relative_path(file_name)
        image_path = _find_image(images_dir, root, relative_path)
        if image_path is None:
            raise DatasetError(
                f"COCO image not found: {file_name!r} (looked in {images_dir} and {root})"
            )

        width = _positive_int(image.get("width"))
        height = _positive_int(image.get("height"))
        if not width or not height:
            width, height = read_image_size(image_path)

        records.append(
            ImageRecord(
                path=image_path,
                relative_path=relative_path,
                width=width,
                height=height,
                annotations=annotations_by_image.get(image_id, []),
            )
        )

    if not records:
        raise DatasetError(f"No images found in COCO file {annotations_path}")

    return Dataset(
        images=records,
        classes=class_names,
        source_format="coco",
        root=root,
        metadata={
            "annotations": str(annotations_path),
            "images": str(images_dir),
            "parse_errors": parse_errors,
        },
    )


def _find_annotations(source: dict, root: Path) -> Path:
    configured = source.get("annotations")
    if configured:
        path = resolve_path(configured, root)
        if path is None or not path.is_file():
            raise DatasetError(f"COCO annotations file not found: {path}")
        return path

    candidates: list[Path] = []
    annotations_dir = root / "annotations"
    if annotations_dir.is_dir():
        candidates.extend(sorted(annotations_dir.glob("*.json")))
    candidates.extend(sorted(root.glob("*.json")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise DatasetError(
        f"No COCO JSON file found under {root}. Set source.annotations to the COCO JSON path."
    )


def _find_image(images_dir: Path, root: Path, relative_path: Path) -> Path | None:
    for candidate in (images_dir / relative_path, root / relative_path):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _positive_int(value) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None
