"""CVAT for images 1.1 XML loader."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path

from ..dataset import Annotation, Dataset, ImageRecord
from ..errors import DatasetError
from ..utils import read_image_size, resolve_path, safe_relative_path


def load(source: dict, root: Path, class_names: list[str] | None) -> Dataset:
    annotations_path = resolve_path(source.get("annotations", "annotations.xml"), root)
    if annotations_path is None or not annotations_path.is_file():
        raise DatasetError(f"CVAT annotations file not found: {annotations_path}")

    images_dir = resolve_path(source.get("images", "images"), root)
    if images_dir is None or not images_dir.is_dir():
        images_dir = root

    try:
        tree = ET.parse(annotations_path)
    except ET.ParseError as error:
        raise DatasetError(f"Could not parse CVAT XML {annotations_path}: {error}") from error

    xml_root = tree.getroot()
    skip_outside = bool(source.get("skip_outside", True))
    allow_unknown_labels = bool(source.get("allow_unknown_labels", False))

    meta_labels = _meta_labels(xml_root)
    box_labels = _box_labels(xml_root)

    if class_names is None:
        class_names = list(meta_labels) if meta_labels else list(box_labels)
    else:
        class_names = list(class_names)

    if not class_names and not box_labels:
        raise DatasetError("No label names found in the CVAT XML")
    if not class_names:
        raise DatasetError("The CVAT XML contains boxes but no class names")

    records: list[ImageRecord] = []
    parse_errors: list[str] = []

    for image_element in xml_root.findall("image"):
        name = image_element.get("name")
        if not name:
            parse_errors.append("CVAT <image> element without a name was skipped")
            continue

        relative_path = safe_relative_path(name)
        image_path = _find_image(images_dir, root, relative_path, name)
        if image_path is None:
            raise DatasetError(
                f"CVAT image not found: {name!r} (looked in {images_dir} and {root})"
            )

        width = _int_or_none(image_element.get("width"))
        height = _int_or_none(image_element.get("height"))
        if not width or not height:
            width, height = read_image_size(image_path)

        annotations: list[Annotation] = []
        for box in image_element.findall("box"):
            if skip_outside and box.get("outside") == "1":
                continue

            label = box.get("label")
            if not label:
                parse_errors.append(f"{name}: box without a label was skipped")
                continue
            if label not in class_names:
                if not allow_unknown_labels:
                    raise DatasetError(
                        f"Unknown CVAT label {label!r} in {name}. "
                        "Add it to source.class_names or set "
                        "source.allow_unknown_labels: true"
                    )
                class_names.append(label)

            try:
                x1 = float(box.get("xtl", ""))
                y1 = float(box.get("ytl", ""))
                x2 = float(box.get("xbr", ""))
                y2 = float(box.get("ybr", ""))
            except ValueError:
                parse_errors.append(f"{name}: box with invalid coordinates was skipped")
                continue

            annotations.append(Annotation(class_names.index(label), x1, y1, x2, y2))

        records.append(
            ImageRecord(
                path=image_path,
                relative_path=relative_path,
                width=width,
                height=height,
                annotations=annotations,
            )
        )

    if not records:
        raise DatasetError(f"No images found in CVAT XML {annotations_path}")

    return Dataset(
        images=records,
        classes=class_names,
        source_format="cvat_xml",
        root=root,
        metadata={
            "annotations": str(annotations_path),
            "images": str(images_dir),
            "parse_errors": parse_errors,
        },
    )


def _meta_labels(xml_root: ET.Element) -> list[str]:
    names: OrderedDict[str, None] = OrderedDict()
    meta = xml_root.find("meta")
    if meta is None:
        return []
    for label in meta.iter("label"):
        name = label.find("name")
        if name is not None and name.text and name.text.strip():
            names.setdefault(name.text.strip(), None)
    return list(names)


def _box_labels(xml_root: ET.Element) -> list[str]:
    names: OrderedDict[str, None] = OrderedDict()
    for box in xml_root.iter("box"):
        label = box.get("label")
        if label:
            names.setdefault(label, None)
    return list(names)


def _find_image(images_dir: Path, root: Path, relative_path: Path, raw_name: str) -> Path | None:
    candidates = [images_dir / relative_path, root / relative_path]
    raw_path = Path(str(raw_name).replace("\\", "/"))
    if raw_path.is_absolute():
        candidates.insert(0, raw_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _int_or_none(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None
