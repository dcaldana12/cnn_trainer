"""Tests for image-list and batch output blocks."""

from __future__ import annotations

import pytest

from cnn_trainer.blocks.output import make_batches, write_image_list
from cnn_trainer.blocks.split import split
from cnn_trainer.errors import BlockError
from cnn_trainer.formats import load_dataset


def _dataset(coco_root):
    return load_dataset(
        {
            "format": "coco",
            "root": str(coco_root),
            "annotations": "annotations/instances_default.json",
            "images": "images",
        },
        coco_root.parent,
    )


def test_write_image_list(coco_root, ctx):
    dataset = _dataset(coco_root)
    split(
        dataset,
        ctx,
        ratios={"train": 0.5, "val": 0.25, "test": 0.25},
        seed=42,
        stratify=False,
    )

    write_image_list(
        dataset,
        ctx,
        path="train.txt",
        split="train",
        prefix="data/images/train",
    )

    lines = (ctx.config_dir / "train.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all(line.startswith("data/images/train/") for line in lines)


def test_write_image_list_absolute(coco_root, ctx):
    dataset = _dataset(coco_root)

    write_image_list(dataset, ctx, path="all.txt", absolute=True)

    lines = (ctx.config_dir / "all.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert all(line.startswith("/") for line in lines)


def test_make_batches_and_zip(coco_root, ctx):
    dataset = _dataset(coco_root)

    make_batches(dataset, ctx, path="batches", batch_size=2)

    destination = ctx.config_dir / "batches"
    assert (destination / "batch_001" / "data.yaml").is_file()
    assert (destination / "batch_002" / "data.yaml").is_file()
    assert (destination / "batch_001.zip").is_file()
    assert (destination / "batch_002.zip").is_file()
    assert len(list((destination / "batch_001" / "images").rglob("*.png"))) == 2


def test_make_batches_filters_split(coco_root, ctx):
    dataset = _dataset(coco_root)
    split(
        dataset,
        ctx,
        ratios={"train": 0.5, "val": 0.25, "test": 0.25},
        seed=42,
        stratify=False,
    )

    make_batches(dataset, ctx, path="train_batches", batch_size=10, split="train")

    destination = ctx.config_dir / "train_batches"
    assert [path.name for path in destination.glob("batch_*") if path.is_dir()] == ["batch_001"]


def test_make_batches_overwrite_protection(coco_root, ctx):
    dataset = _dataset(coco_root)
    make_batches(dataset, ctx, path="batches", batch_size=10)

    with pytest.raises(BlockError, match="not empty"):
        make_batches(dataset, ctx, path="batches", batch_size=10)


def test_write_image_list_rejects_unknown_split(coco_root, ctx):
    dataset = _dataset(coco_root)
    with pytest.raises(BlockError, match="split must be one of"):
        write_image_list(dataset, ctx, path="x.txt", split="training")
