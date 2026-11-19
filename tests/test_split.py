"""Tests for train/validation/test splitting."""

from __future__ import annotations

import copy
from collections import Counter

import pytest

from cnn_trainer.blocks.split import split
from cnn_trainer.errors import BlockError
from tests.helpers import make_records


def _counts(dataset):
    return Counter(record.split for record in dataset.images)


def test_split_exact_counts(ctx):
    dataset = make_records(100, num_classes=3)

    split(
        dataset,
        ctx,
        ratios={"train": 0.7, "val": 0.2, "test": 0.1},
        seed=1,
        stratify=False,
    )

    counts = _counts(dataset)
    assert counts == {"train": 70, "val": 20, "test": 10}
    assert all(record.split is not None for record in dataset.images)


def test_split_is_reproducible(ctx):
    first = make_records(50, num_classes=2)
    second = copy.deepcopy(first)

    split(first, ctx, ratios={"train": 0.6, "val": 0.2, "test": 0.2}, seed=7, stratify=False)
    split(second, ctx, ratios={"train": 0.6, "val": 0.2, "test": 0.2}, seed=7, stratify=False)

    assignment_first = [(record.relative_path, record.split) for record in first.images]
    assignment_second = [(record.relative_path, record.split) for record in second.images]
    assert assignment_first == assignment_second


def test_split_changes_with_seed(ctx):
    first = make_records(50, num_classes=2)
    second = copy.deepcopy(first)

    split(first, ctx, ratios={"train": 0.6, "val": 0.2, "test": 0.2}, seed=1, stratify=False)
    split(second, ctx, ratios={"train": 0.6, "val": 0.2, "test": 0.2}, seed=2, stratify=False)

    assignment_first = {(record.relative_path, record.split) for record in first.images}
    assignment_second = {(record.relative_path, record.split) for record in second.images}
    assert assignment_first != assignment_second


def test_stratified_split_keeps_classes_in_each_split(ctx):
    dataset = make_records(100, num_classes=2)

    split(
        dataset,
        ctx,
        ratios={"train": 0.6, "val": 0.2, "test": 0.2},
        seed=3,
        stratify=True,
    )

    assert _counts(dataset) == {"train": 60, "val": 20, "test": 20}
    for split_name in ("train", "val", "test"):
        class_ids = {
            annotation.class_id
            for record in dataset.images
            if record.split == split_name
            for annotation in record.annotations
        }
        assert class_ids == {0, 1}


def test_split_normalizes_ratios(ctx):
    dataset = make_records(100, num_classes=2)

    split(dataset, ctx, ratios={"train": 7, "val": 2, "test": 1}, seed=4, stratify=False)

    assert _counts(dataset) == {"train": 70, "val": 20, "test": 10}


def test_split_can_skip_a_split(ctx):
    dataset = make_records(10, num_classes=2)

    split(dataset, ctx, ratios={"train": 0.8, "val": 0.2, "test": 0}, seed=5, stratify=False)

    counts = _counts(dataset)
    assert counts["test"] == 0
    assert counts["train"] == 8
    assert counts["val"] == 2


def test_split_overwrite_protection(ctx):
    dataset = make_records(10, num_classes=2)
    split(dataset, ctx, ratios={"train": 0.8, "val": 0.2}, seed=5, stratify=False)

    with pytest.raises(BlockError, match="already has split"):
        split(dataset, ctx, ratios={"train": 0.8, "val": 0.2}, seed=5, overwrite=False)


def test_split_rejects_invalid_ratios(ctx):
    dataset = make_records(10, num_classes=2)

    with pytest.raises(BlockError, match="non-negative"):
        split(dataset, ctx, ratios={"train": 1.2, "val": -0.2})

    with pytest.raises(BlockError, match="Unknown split"):
        split(dataset, ctx, ratios={"training": 1.0})
