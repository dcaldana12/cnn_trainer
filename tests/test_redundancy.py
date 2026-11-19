"""Tests for near-duplicate detection and image removal."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from cnn_trainer.blocks.audit import audit_dataset  # noqa: E402
from cnn_trainer.blocks.redundancy import (  # noqa: E402
    find_redundant_images,
    remove_flagged_images,
)
from cnn_trainer.dataset import Annotation, Dataset, ImageRecord  # noqa: E402
from cnn_trainer.errors import BlockError  # noqa: E402
from tests.helpers import write_png  # noqa: E402


def _write_noise(path: Path, seed: int, *, size: int = 64, tweak: bool = False) -> None:
    rng = random.Random(seed)
    data = bytearray(rng.randrange(256) for _ in range(size * size))
    if tweak:
        data[0] ^= 1
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.frombytes("L", (size, size), bytes(data)).save(path)


def _dataset(tmp_path: Path, items: list[dict]) -> Dataset:
    images = []
    for item in items:
        path = tmp_path / item["name"]
        _write_noise(path, item["seed"], tweak=item.get("tweak", False))
        images.append(
            ImageRecord(
                path=path,
                relative_path=Path(item["name"]),
                width=64,
                height=64,
                annotations=[
                    Annotation(0, 10.0, 10.0, 30.0, 30.0) for _ in range(item.get("boxes", 0))
                ],
                split=item.get("split"),
            )
        )
    return Dataset(
        images=images,
        classes=["object"],
        source_format="test",
        root=tmp_path,
    )


def test_find_flags_near_duplicates_and_keeps_annotated(tmp_path, ctx):
    dataset = _dataset(
        tmp_path,
        [
            {"name": "a.png", "seed": 1},
            {"name": "b.png", "seed": 1, "tweak": True, "boxes": 1},
            {"name": "c.png", "seed": 2},
        ],
    )

    find_redundant_images(dataset, ctx, min_contrast=0.0, similarity_threshold=2)

    assert dataset.metadata["image_flags"]["redundant"] == ["a.png"]
    report = json.loads((ctx.workdir / "redundancy.json").read_text(encoding="utf-8"))
    assert report["counts"] == {"duplicate": 1, "blank": 0}
    assert report["duplicate_groups"][0]["keeper"] == "b.png"
    assert report["duplicate_groups"][0]["size"] == 2
    candidates = (ctx.workdir / "candidates.txt").read_text(encoding="utf-8")
    assert candidates == "a.png\n"


def test_find_flags_blank_images_but_not_annotated_ones(tmp_path, ctx):
    dataset = _dataset(
        tmp_path,
        [
            {"name": "noise.png", "seed": 3},
            {"name": "labeled.png", "seed": 4, "boxes": 1},
        ],
    )
    write_png(tmp_path / "blank.png", 64, 64, (20, 20, 20))
    dataset.images.append(
        ImageRecord(
            path=tmp_path / "blank.png",
            relative_path=Path("blank.png"),
            width=64,
            height=64,
        )
    )

    find_redundant_images(dataset, ctx, min_contrast=5.0, similarity_threshold=0)

    candidates = dataset.metadata["redundancy"]["candidates"]
    by_path = {candidate["relative_path"]: candidate for candidate in candidates}
    assert "blank.png" in by_path
    assert by_path["blank.png"]["reason"] == "blank"
    assert "labeled.png" not in by_path


def test_find_can_respect_splits(tmp_path, ctx):
    dataset = _dataset(
        tmp_path,
        [
            {"name": "a.png", "seed": 1, "split": "train"},
            {"name": "b.png", "seed": 1, "split": "val"},
        ],
    )

    find_redundant_images(
        dataset, ctx, min_contrast=0.0, similarity_threshold=0, respect_splits=True
    )
    assert dataset.metadata["image_flags"]["redundant"] == []

    find_redundant_images(
        dataset, ctx, min_contrast=0.0, similarity_threshold=0, respect_splits=False
    )
    assert len(dataset.metadata["image_flags"]["redundant"]) == 1


def test_find_can_protect_test_split(tmp_path, ctx):
    dataset = _dataset(
        tmp_path,
        [
            {"name": "a.png", "seed": 1, "boxes": 5, "split": "val"},
            {"name": "b.png", "seed": 1, "split": "test"},
        ],
    )

    find_redundant_images(
        dataset,
        ctx,
        min_contrast=0.0,
        similarity_threshold=0,
        protect_splits=["test"],
    )

    assert dataset.metadata["image_flags"]["redundant"] == ["a.png"]


def test_find_reports_cross_split_duplicates(tmp_path, ctx):
    dataset = _dataset(
        tmp_path,
        [
            {"name": "a.png", "seed": 1, "split": "train"},
            {"name": "b.png", "seed": 1, "split": "val"},
        ],
    )

    find_redundant_images(dataset, ctx, min_contrast=0.0, similarity_threshold=0)

    report = dataset.metadata["redundancy"]
    assert report["cross_split_groups"] == 1
    assert report["duplicate_groups"][0]["cross_split"] is True

    audit_dataset(dataset, ctx, min_instances_per_class=1)
    audit = json.loads((ctx.workdir / "audit.json").read_text(encoding="utf-8"))
    assert any("span different splits" in tip for tip in audit["tips"])


def test_find_protects_every_image_in_a_protected_split(tmp_path, ctx):
    dataset = _dataset(
        tmp_path,
        [
            {"name": "a.png", "seed": 1, "split": "test"},
            {"name": "b.png", "seed": 1, "split": "test"},
        ],
    )

    find_redundant_images(
        dataset,
        ctx,
        min_contrast=0.0,
        similarity_threshold=0,
        protect_splits=["test"],
    )

    assert dataset.metadata["image_flags"]["redundant"] == []


def test_remove_flagged_images_updates_dataset_and_report(tmp_path, ctx):
    dataset = _dataset(
        tmp_path,
        [
            {"name": "a.png", "seed": 1},
            {"name": "b.png", "seed": 1, "boxes": 1},
            {"name": "c.png", "seed": 2},
        ],
    )
    find_redundant_images(dataset, ctx, min_contrast=0.0, similarity_threshold=0)

    remove_flagged_images(dataset, ctx)

    assert [record.relative_path.name for record in dataset.images] == [
        "b.png",
        "c.png",
    ]
    assert "redundant" not in dataset.metadata["image_flags"]
    assert dataset.metadata["redundancy"]["candidates"] == []
    assert dataset.metadata["redundancy"]["cross_split_groups"] == 0
    report = json.loads((ctx.workdir / "removed.json").read_text(encoding="utf-8"))
    assert report["removed"] == 1
    assert report["reason_counts"] == {"duplicate": 1}


def test_remove_dry_run_keeps_images(tmp_path, ctx):
    dataset = _dataset(
        tmp_path,
        [
            {"name": "a.png", "seed": 1},
            {"name": "b.png", "seed": 1, "boxes": 1},
        ],
    )
    find_redundant_images(dataset, ctx, min_contrast=0.0, similarity_threshold=0)

    remove_flagged_images(dataset, ctx, dry_run=True)

    assert len(dataset.images) == 2
    report = json.loads((ctx.workdir / "removed.json").read_text(encoding="utf-8"))
    assert report["removed"] == 0
    assert report["dry_run"] is True


def test_remove_refuses_too_many_images(tmp_path, ctx):
    dataset = _dataset(
        tmp_path,
        [
            {"name": "a.png", "seed": 1, "boxes": 1},
            {"name": "b.png", "seed": 1, "boxes": 1},
            {"name": "c.png", "seed": 1, "boxes": 1},
            {"name": "d.png", "seed": 1, "boxes": 1},
        ],
    )
    find_redundant_images(dataset, ctx, min_contrast=0.0, similarity_threshold=0)

    with pytest.raises(BlockError, match="max_fraction"):
        remove_flagged_images(dataset, ctx, max_fraction=0.5)


def test_remove_filters_candidates_by_reason(tmp_path, ctx):
    dataset = _dataset(
        tmp_path,
        [
            {"name": "a.png", "seed": 1},
            {"name": "b.png", "seed": 1, "boxes": 1},
            {"name": "c.png", "seed": 2},
        ],
    )
    find_redundant_images(dataset, ctx, min_contrast=0.0, similarity_threshold=0)

    remove_flagged_images(dataset, ctx, reasons=["blank"])
    assert len(dataset.images) == 3

    remove_flagged_images(dataset, ctx, reasons=["duplicate"])
    assert len(dataset.images) == 2


def test_remove_accepts_explicit_paths(tmp_path, ctx):
    dataset = _dataset(
        tmp_path,
        [
            {"name": "a.png", "seed": 1},
            {"name": "b.png", "seed": 2},
            {"name": "c.png", "seed": 3},
        ],
    )

    remove_flagged_images(dataset, ctx, paths=["a.png"])

    assert [record.relative_path.name for record in dataset.images] == [
        "b.png",
        "c.png",
    ]
    report = json.loads((ctx.workdir / "removed.json").read_text(encoding="utf-8"))
    assert report["reason_counts"] == {"explicit": 1}


def test_remove_requires_a_flag_or_paths(tmp_path, ctx):
    dataset = _dataset(tmp_path, [{"name": "a.png", "seed": 1}])

    with pytest.raises(BlockError, match="No images are flagged"):
        remove_flagged_images(dataset, ctx)


def test_find_requires_pillow(tmp_path, ctx, monkeypatch):
    dataset = _dataset(tmp_path, [{"name": "a.png", "seed": 1}])
    monkeypatch.setitem(sys.modules, "PIL", None)

    with pytest.raises(BlockError, match="Pillow"):
        find_redundant_images(dataset, ctx, min_contrast=0.0)


def test_find_rejects_invalid_threshold(tmp_path, ctx):
    dataset = _dataset(tmp_path, [{"name": "a.png", "seed": 1}])

    with pytest.raises(BlockError, match="similarity_threshold"):
        find_redundant_images(dataset, ctx, similarity_threshold=9)
    with pytest.raises(BlockError, match="hash_size"):
        find_redundant_images(dataset, ctx, hash_size=4)
