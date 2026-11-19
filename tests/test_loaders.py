"""Tests for the CVAT, COCO and YOLO source adapters."""

from __future__ import annotations

import pytest

from cnn_trainer.errors import DatasetError
from cnn_trainer.formats import load_dataset


def _coco_source(root):
    return {
        "format": "coco",
        "root": str(root),
        "annotations": "annotations/instances_default.json",
        "images": "images",
    }


def test_load_coco(coco_root):
    dataset = load_dataset(_coco_source(coco_root), coco_root.parent)

    assert dataset.source_format == "coco"
    assert dataset.classes == ["cat", "dog"]
    assert dataset.metadata["parse_errors"] == []

    records = {record.relative_path.name: record for record in dataset.images}
    assert set(records) == {"a.png", "b.png", "c.png", "d.png"}

    a = records["a.png"]
    assert len(a.annotations) == 2
    assert {annotation.class_id for annotation in a.annotations} == {0, 1}
    cat_box = next(annotation for annotation in a.annotations if annotation.class_id == 0)
    assert (cat_box.x1, cat_box.y1, cat_box.x2, cat_box.y2) == (10, 10, 30, 30)

    assert len(records["b.png"].annotations) == 1
    assert records["c.png"].annotations == []
    assert len(records["d.png"].annotations) == 1  # crowd annotation skipped


def test_coco_class_names_override(coco_root):
    source = {**_coco_source(coco_root), "class_names": ["dog", "cat"]}
    dataset = load_dataset(source, coco_root.parent)

    assert dataset.classes == ["dog", "cat"]
    a = next(record for record in dataset.images if record.relative_path.name == "a.png")
    cat_box = next(annotation for annotation in a.annotations if annotation.class_id == 1)
    assert (cat_box.x1, cat_box.y1) == (10, 10)


def test_coco_keeps_crowd_when_requested(coco_root):
    source = {**_coco_source(coco_root), "skip_crowd": False}
    dataset = load_dataset(source, coco_root.parent)
    d = next(record for record in dataset.images if record.relative_path.name == "d.png")
    assert len(d.annotations) == 2


def test_coco_missing_annotation_file(coco_root):
    source = {**_coco_source(coco_root), "annotations": "missing.json"}
    with pytest.raises(DatasetError, match="COCO annotations file not found"):
        load_dataset(source, coco_root.parent)


def test_coco_rejects_missing_category_name(coco_root):
    source = {**_coco_source(coco_root), "class_names": ["cat"]}
    with pytest.raises(DatasetError, match="missing from source.class_names"):
        load_dataset(source, coco_root.parent)


def test_load_cvat(cvat_root):
    source = {
        "format": "cvat_xml",
        "root": str(cvat_root),
        "annotations": "annotations.xml",
        "images": "images",
    }
    dataset = load_dataset(source, cvat_root.parent)

    assert dataset.source_format == "cvat_xml"
    assert dataset.classes == ["car", "person"]
    records = {record.relative_path.name: record for record in dataset.images}
    assert len(records["a.png"].annotations) == 2
    assert {annotation.class_id for annotation in records["a.png"].annotations} == {0, 1}
    assert records["b.png"].annotations == []


def test_cvat_keeps_outside_boxes_when_requested(cvat_root):
    source = {
        "format": "cvat_xml",
        "root": str(cvat_root),
        "annotations": "annotations.xml",
        "images": "images",
        "skip_outside": False,
    }
    dataset = load_dataset(source, cvat_root.parent)
    a = next(record for record in dataset.images if record.relative_path.name == "a.png")
    assert len(a.annotations) == 3


def test_load_yolo(yolo_root):
    source = {"format": "yolo", "root": str(yolo_root)}
    dataset = load_dataset(source, yolo_root.parent)

    assert dataset.source_format == "yolo"
    assert dataset.classes == ["cat", "dog"]

    records = {record.relative_path.as_posix(): record for record in dataset.images}
    assert set(records) == {"train/a.png", "val/b.png", "val/c.png"}
    assert records["train/a.png"].split == "train"
    assert records["val/b.png"].split == "val"
    assert records["val/c.png"].has_label is False

    a_box = records["train/a.png"].annotations[0]
    assert (a_box.x1, a_box.y1, a_box.x2, a_box.y2) == (40, 40, 60, 60)

    b_box = records["val/b.png"].annotations[0]
    assert b_box.class_id == 1
    assert (b_box.x1, b_box.y1, b_box.x2, b_box.y2) == (60, 40, 140, 60)

    assert len(dataset.orphans) == 1
    assert dataset.orphans[0].name == "orphan.txt"


def test_yolo_custom_images_dir_mirrors_labels(yolo_root):
    source = {"format": "yolo", "root": str(yolo_root), "images": "images/train"}
    dataset = load_dataset(source, yolo_root.parent)

    assert len(dataset.images) == 1
    assert dataset.images[0].relative_path.as_posix() == "a.png"
    assert dataset.images[0].has_label is True


def test_yolo_infers_class_names_without_data_yaml(yolo_root):
    (yolo_root / "data.yaml").unlink()
    dataset = load_dataset({"format": "yolo", "root": str(yolo_root)}, yolo_root.parent)
    assert dataset.classes == ["class_0", "class_1"]


def test_yolo_uses_nc_when_names_are_absent(yolo_root):
    (yolo_root / "data.yaml").write_text(
        "nc: 3\ntrain: images/train\nval: images/val\n", encoding="utf-8"
    )
    dataset = load_dataset({"format": "yolo", "root": str(yolo_root)}, yolo_root.parent)
    assert dataset.classes == ["class_0", "class_1", "class_2"]


def test_yolo_missing_labels_directory(yolo_root):
    (yolo_root / "labels").rename(yolo_root / "labels_old")
    dataset = load_dataset({"format": "yolo", "root": str(yolo_root)}, yolo_root.parent)

    assert all(record.has_label is False for record in dataset.images)
    assert any(
        "Labels directory not found" in message for message in dataset.metadata["parse_errors"]
    )


def test_yolo_reports_malformed_lines(yolo_root):
    (yolo_root / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2\n", encoding="utf-8")
    dataset = load_dataset({"format": "yolo", "root": str(yolo_root)}, yolo_root.parent)

    assert dataset.images[0].annotations == [] or len(dataset.images[0].annotations) == 0
    assert any("expected 5 values" in message for message in dataset.metadata["parse_errors"])


def test_yolo_rejects_unknown_class_id(yolo_root):
    (yolo_root / "labels" / "train" / "a.txt").write_text("2 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="class id 2"):
        load_dataset({"format": "yolo", "root": str(yolo_root)}, yolo_root.parent)
