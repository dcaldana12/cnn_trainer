"""Tests for the dataset audit block."""

from __future__ import annotations

import json

import pytest

from cnn_trainer.blocks.audit import audit_dataset
from cnn_trainer.dataset import Annotation
from cnn_trainer.errors import BlockError
from tests.helpers import make_records


def test_audit_reports_balance_split_gaps_and_quality(ctx):
    dataset = make_records(6, num_classes=3)
    dataset.images[0].annotations = []
    dataset.images[0].has_label = False
    for _ in range(20):
        dataset.images[1].annotations.append(Annotation(0, 10.0, 10.0, 30.0, 30.0))
    for index, split in enumerate(("train", "train", "train", "train", "val", "test")):
        dataset.images[index].split = split

    audit_dataset(
        dataset,
        ctx,
        min_instances_per_class=3,
        min_images_per_class=2,
        max_imbalance_ratio=5.0,
    )

    report = json.loads((ctx.workdir / "audit.json").read_text(encoding="utf-8"))
    assert report["images"] == 6
    assert report["empty_images"] == 1
    assert report["missing_labels"] == 1
    assert report["class_instances"]["class_0"] == 21
    assert report["class_instances"]["class_1"] == 2
    assert "train" in report["splits"]
    assert report["unassigned"] == 0

    tips = "\n".join(report["tips"])
    assert "Rare class" in tips
    assert "Class imbalance" in tips
    assert "no label file" in tips
    assert "no instances in val" in tips

    markdown = (ctx.workdir / "audit.md").read_text(encoding="utf-8")
    assert "| class_0 |" in markdown
    assert dataset.metadata["audit"]["images"] == 6


def test_audit_flags_tiny_and_border_boxes(ctx):
    dataset = make_records(2, num_classes=1, width=100, height=100)
    dataset.images[0].annotations = [Annotation(0, 0.0, 0.0, 1.0, 1.0)]

    audit_dataset(dataset, ctx, min_instances_per_class=10)

    report = json.loads((ctx.workdir / "audit.json").read_text(encoding="utf-8"))
    assert report["box_area_ratio"]["tiny"] == 1
    assert report["box_area_ratio"]["near_border"] == 1
    tips = "\n".join(report["tips"])
    assert "cover less than" in tips
    assert "touch the image border" in tips


def test_audit_warns_about_missing_validation_split(ctx):
    dataset = make_records(3, num_classes=1)
    for record in dataset.images:
        record.split = "train"

    audit_dataset(dataset, ctx, min_instances_per_class=1)

    report = json.loads((ctx.workdir / "audit.json").read_text(encoding="utf-8"))
    tips = "\n".join(report["tips"])
    assert "No validation images" in tips


def test_audit_uses_redundancy_metadata(ctx):
    dataset = make_records(4, num_classes=1)
    dataset.metadata["redundancy"] = {
        "candidates": [
            {"relative_path": "1.png", "reason": "duplicate"},
            {"relative_path": "2.png", "reason": "blank"},
        ],
        "counts": {"duplicate": 1, "blank": 1},
    }

    audit_dataset(dataset, ctx, min_instances_per_class=1)

    report = json.loads((ctx.workdir / "audit.json").read_text(encoding="utf-8"))
    assert report["redundancy"]["candidates"] == 2
    assert any("remove_flagged_images" in tip for tip in report["tips"])


def test_audit_rejects_invalid_options(ctx):
    dataset = make_records(2)

    with pytest.raises(BlockError, match="tiny_box_area"):
        audit_dataset(dataset, ctx, tiny_box_area=2.0)
    with pytest.raises(BlockError, match="min_image_side"):
        audit_dataset(dataset, ctx, min_image_side=-1)


def test_audit_rejects_empty_dataset(ctx):
    dataset = make_records(0)

    with pytest.raises(BlockError, match="empty dataset"):
        audit_dataset(dataset, ctx)
