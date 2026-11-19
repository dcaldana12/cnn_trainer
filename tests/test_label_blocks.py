"""Tests for label editing and validation blocks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cnn_trainer.blocks.labels import create_missing_labels, remap_classes, remove_class
from cnn_trainer.blocks.stats import stats
from cnn_trainer.blocks.validate import validate
from cnn_trainer.errors import BlockError
from tests.helpers import make_records


def test_create_missing_labels(ctx):
    dataset = make_records(2)
    dataset.images[0].has_label = False

    create_missing_labels(dataset, ctx)

    assert all(record.has_label for record in dataset.images)


def test_remap_merges_into_existing_class(ctx):
    dataset = make_records(3, num_classes=3)

    remap_classes(dataset, ctx, mapping={2: 0})

    assert dataset.classes == ["class_0", "class_1", "class_2"]
    assert dataset.images[2].annotations[0].class_id == 0


def test_remap_renames_in_place(ctx):
    dataset = make_records(3, num_classes=3)

    remap_classes(dataset, ctx, mapping={2: "vehicle"})

    assert dataset.classes == ["class_0", "class_1", "vehicle"]
    assert dataset.images[2].annotations[0].class_id == 2


def test_remap_by_class_name(ctx):
    dataset = make_records(3, num_classes=3)

    remap_classes(dataset, ctx, mapping={"class_0": "class_1"})

    assert dataset.images[0].annotations[0].class_id == 1


def test_remap_accepts_quoted_class_id(ctx):
    dataset = make_records(3, num_classes=3)

    remap_classes(dataset, ctx, mapping={2: "0"})

    assert dataset.images[2].annotations[0].class_id == 0


def test_remap_drops_annotations(ctx):
    dataset = make_records(3, num_classes=3)

    remap_classes(dataset, ctx, mapping={1: None})

    assert dataset.images[1].annotations == []
    assert dataset.classes == ["class_0", "class_1", "class_2"]


def test_remap_prunes_unused_classes(ctx):
    dataset = make_records(3, num_classes=3)

    remap_classes(dataset, ctx, mapping={1: None}, prune_unused=True)

    assert dataset.classes == ["class_0", "class_2"]
    assert dataset.images[2].annotations[0].class_id == 1


def test_remove_class_and_prune(ctx):
    dataset = make_records(3, num_classes=3)

    remove_class(dataset, ctx, classes=[1], prune_unused=True)

    assert dataset.classes == ["class_0", "class_2"]
    assert dataset.images[1].annotations == []


def test_duplicate_mapping_is_rejected(ctx):
    dataset = make_records(1, num_classes=2)
    with pytest.raises(BlockError, match="mapped more than once"):
        remap_classes(dataset, ctx, mapping={0: 1, "class_0": None})


def test_invalid_mapping_key_is_rejected(ctx):
    dataset = make_records(1, num_classes=2)
    with pytest.raises(BlockError, match="outside"):
        remap_classes(dataset, ctx, mapping={99: 0})


def test_validate_detects_missing_labels(ctx):
    dataset = make_records(1)
    dataset.images[0].has_label = False

    with pytest.raises(BlockError, match="validation failed"):
        validate(dataset, ctx)

    create_missing_labels(dataset, ctx)
    validate(dataset, ctx)


def test_validate_detects_out_of_bounds_box(ctx):
    dataset = make_records(1)
    dataset.images[0].annotations[0].x2 = 200

    with pytest.raises(BlockError, match="validation failed"):
        validate(dataset, ctx)


def test_validate_non_strict_writes_report(ctx):
    dataset = make_records(1)
    dataset.images[0].annotations[0].x2 = 200

    validate(dataset, ctx, strict=False)

    report = json.loads((ctx.workdir / "validation.json").read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert any("outside image bounds" in message for message in report["errors"])


def test_validate_allows_same_stem_in_different_splits(ctx):
    dataset = make_records(2)
    dataset.images[0].relative_path = Path("train/a.png")
    dataset.images[0].split = "train"
    dataset.images[1].relative_path = Path("val/a.png")
    dataset.images[1].split = "val"

    validate(dataset, ctx)


def test_validate_pads_class_names(ctx):
    dataset = make_records(1, num_classes=2)

    validate(dataset, ctx, num_classes=3)

    assert dataset.classes == ["class_0", "class_1", "class_2"]


def test_stats_writes_json(ctx):
    dataset = make_records(3, num_classes=3)

    stats(dataset, ctx)

    payload = json.loads((ctx.workdir / "stats.json").read_text(encoding="utf-8"))
    assert payload["stats"]["images"] == 3
    assert payload["stats"]["annotations"] == 3
    assert payload["classes"] == ["class_0", "class_1", "class_2"]
