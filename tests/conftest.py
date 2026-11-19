"""Pytest fixtures with synthetic datasets."""

from __future__ import annotations

import json

import pytest

from .helpers import make_context, write_png


@pytest.fixture
def ctx(tmp_path):
    return make_context(tmp_path)


@pytest.fixture
def coco_root(tmp_path):
    root = tmp_path / "coco"
    images = root / "images"
    write_png(images / "a.png", 100, 100, (200, 10, 10))
    write_png(images / "b.png", 200, 100, (10, 200, 10))
    write_png(images / "c.png", 100, 50, (10, 10, 200))
    write_png(images / "d.png", 100, 100, (120, 120, 10))

    payload = {
        "images": [
            {"id": 1, "file_name": "a.png", "width": 100, "height": 100},
            {"id": 2, "file_name": "b.png", "width": 200, "height": 100},
            {"id": 3, "file_name": "c.png", "width": 100, "height": 50},
            {"id": 4, "file_name": "d.png", "width": 100, "height": 100},
        ],
        "categories": [
            {"id": 1, "name": "cat"},
            {"id": 2, "name": "dog"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [50, 50, 30, 20], "iscrowd": 0},
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [0, 0, 100, 50], "iscrowd": 0},
            {"id": 4, "image_id": 4, "category_id": 2, "bbox": [20, 20, 20, 20], "iscrowd": 0},
            {"id": 5, "image_id": 4, "category_id": 2, "bbox": [0, 0, 10, 10], "iscrowd": 1},
        ],
    }
    annotations_dir = root / "annotations"
    annotations_dir.mkdir(parents=True)
    (annotations_dir / "instances_default.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return root


@pytest.fixture
def cvat_root(tmp_path):
    root = tmp_path / "cvat"
    images = root / "images"
    write_png(images / "a.png", 100, 100)
    write_png(images / "b.png", 100, 100)

    xml = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <labels>
        <label><name>car</name></label>
        <label><name>person</name></label>
      </labels>
    </task>
  </meta>
  <image id="0" name="a.png" width="100" height="100">
    <box label="person" xtl="10" ytl="10" xbr="20" ybr="20" outside="0"/>
    <box label="car" xtl="30" ytl="30" xbr="50" ybr="40" outside="0"/>
    <box label="person" xtl="0" ytl="0" xbr="10" ybr="10" outside="1"/>
  </image>
  <image id="1" name="b.png" width="100" height="100"></image>
</annotations>
"""
    (root / "annotations.xml").write_text(xml, encoding="utf-8")
    return root


@pytest.fixture
def yolo_root(tmp_path):
    root = tmp_path / "yolo"
    write_png(root / "images" / "train" / "a.png", 100, 100)
    write_png(root / "images" / "val" / "b.png", 200, 100)
    write_png(root / "images" / "val" / "c.png", 100, 100)

    (root / "labels" / "train").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    (root / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (root / "labels" / "val" / "b.txt").write_text("1 0.5 0.5 0.4 0.2\n", encoding="utf-8")
    (root / "labels" / "train" / "orphan.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    (root / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: cat\n  1: dog\n",
        encoding="utf-8",
    )
    return root
