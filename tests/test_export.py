"""Tests for YOLO export and the training block's dependency handling."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from cnn_trainer.blocks.export import export_yolo, write_yolo_dataset
from cnn_trainer.blocks.split import split
from cnn_trainer.blocks.train import train
from cnn_trainer.dataset import Annotation, Dataset, ImageRecord
from cnn_trainer.errors import BlockError
from cnn_trainer.formats import load_dataset


def _coco_dataset(coco_root):
    return load_dataset(
        {
            "format": "coco",
            "root": str(coco_root),
            "annotations": "annotations/instances_default.json",
            "images": "images",
        },
        coco_root.parent,
    )


def _split(dataset, ctx):
    split(
        dataset,
        ctx,
        ratios={"train": 0.5, "val": 0.25, "test": 0.25},
        seed=42,
        stratify=False,
    )


def test_export_layout_and_metadata(coco_root, ctx, tmp_path):
    dataset = _coco_dataset(coco_root)
    _split(dataset, ctx)

    destination = tmp_path / "export"
    data_path = write_yolo_dataset(dataset, destination)

    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    assert "path" not in data
    assert data["nc"] == 2
    assert data["names"] == {0: "cat", 1: "dog"}
    assert data["train"] == "images/train"
    assert data["val"] == "images/val"
    assert data["test"] == "images/test"

    assert len(list((destination / "images").rglob("*.png"))) == 4
    assert len(list((destination / "labels").rglob("*.txt"))) == 4

    a_images = list((destination / "images").glob("*/a.png"))
    assert len(a_images) == 1
    split_name = a_images[0].parent.name
    label_lines = (
        (destination / "labels" / split_name / "a.txt").read_text(encoding="utf-8").splitlines()
    )
    assert len(label_lines) == 2
    assert {line.split()[0] for line in label_lines} == {"0", "1"}

    c_images = list((destination / "images").glob("*/c.png"))
    assert len(c_images) == 1
    c_label = destination / "labels" / c_images[0].parent.name / "c.txt"
    assert c_label.read_text(encoding="utf-8") == ""


def test_export_can_pin_absolute_path(coco_root, tmp_path):
    dataset = _coco_dataset(coco_root)
    destination = tmp_path / "export"

    data_path = write_yolo_dataset(dataset, destination, absolute_path=True)

    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    assert data["path"] == str(destination.resolve())


def test_export_refuses_non_empty_directory(coco_root, ctx, tmp_path):
    dataset = _coco_dataset(coco_root)
    destination = tmp_path / "export"
    write_yolo_dataset(dataset, destination)

    with pytest.raises(BlockError, match="not empty"):
        write_yolo_dataset(dataset, destination)

    write_yolo_dataset(dataset, destination, overwrite=True)


def test_export_rejects_stem_collisions(coco_root, tmp_path):
    image = coco_root / "images" / "a.png"
    dataset = Dataset(
        images=[
            ImageRecord(image, Path("a.png"), 100, 100),
            ImageRecord(image, Path("a.jpg"), 100, 100),
        ],
        classes=["cat"],
        source_format="test",
        root=coco_root,
    )

    with pytest.raises(BlockError, match="same YOLO label"):
        write_yolo_dataset(dataset, tmp_path / "export")


def test_export_rejects_out_of_bounds_but_can_clamp(coco_root, tmp_path):
    image = coco_root / "images" / "a.png"
    dataset = Dataset(
        images=[
            ImageRecord(
                image,
                Path("a.png"),
                100,
                100,
                [Annotation(0, 90, 10, 110, 20)],
            )
        ],
        classes=["cat"],
        source_format="test",
        root=coco_root,
    )

    with pytest.raises(BlockError, match="outside the image"):
        write_yolo_dataset(dataset, tmp_path / "export")

    write_yolo_dataset(dataset, tmp_path / "export", overwrite=True, clamp=True)
    line = (tmp_path / "export" / "labels" / "train" / "a.txt").read_text(encoding="utf-8").strip()
    _, x_center, _, width, _ = line.split()
    assert float(x_center) == pytest.approx(0.95)
    assert float(width) == pytest.approx(0.10)


def test_export_symlink_mode(coco_root, tmp_path):
    dataset = _coco_dataset(coco_root)
    destination = tmp_path / "export"

    write_yolo_dataset(dataset, destination, mode="symlink")

    image = next((destination / "images").rglob("*.png"))
    assert image.is_symlink()


def test_export_yolo_block_resolves_relative_path(coco_root, ctx):
    dataset = _coco_dataset(coco_root)

    export_yolo(dataset, ctx, path="build/yolo")

    assert (ctx.config_dir / "build" / "yolo" / "data.yaml").is_file()


def test_train_requires_validation_split(coco_root, ctx):
    dataset = _coco_dataset(coco_root)

    with pytest.raises(BlockError, match="validation split"):
        train(dataset, ctx, epochs=1)


def test_train_reports_missing_ultralytics(coco_root, ctx, monkeypatch):
    dataset = _coco_dataset(coco_root)
    _split(dataset, ctx)
    monkeypatch.setitem(sys.modules, "ultralytics", None)

    with pytest.raises(BlockError, match="Ultralytics is not installed"):
        train(dataset, ctx, epochs=1)


def test_train_cfg_preserves_backend_settings_and_allows_explicit_overrides(
    coco_root, ctx, monkeypatch
):
    dataset = _coco_dataset(coco_root)
    config = ctx.config_dir / "configs"
    config.mkdir()
    cfg_path = config / "training.yaml"
    cfg_path.write_text(
        "model: weights.pt\ndata: data.yaml\nepochs: 30\nimgsz: 320\nname: custom\n"
    )
    (config / "data.yaml").write_text("train: images/train\nval: images/val\n")
    (config / "weights.pt").touch()
    captured = {}
    output = ctx.workdir / "backend-assigned-run"

    class FakeYOLO:
        def __init__(self, model):
            captured["model"] = model
            self.trainer = SimpleNamespace(save_dir=output)

        def train(self, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    train(dataset, ctx, extra={"cfg": "configs/training.yaml", "batch": 4, "device": 0})

    assert captured == {
        "model": str(config / "weights.pt"),
        "kwargs": {
            "cfg": str(cfg_path),
            "data": str(config / "data.yaml"),
            "batch": 4,
            "device": 0,
        },
    }
    assert dataset.metadata["training_results"] == str(output)
    assert not (ctx.workdir / "training_dataset").exists()

    train(dataset, ctx, epochs=2, extra={"cfg": str(cfg_path), "epochs": 6, "model": "other.pt"})
    assert captured["model"] == "other.pt"
    assert captured["kwargs"]["epochs"] == 2
