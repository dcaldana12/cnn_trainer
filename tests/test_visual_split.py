"""Visual split integration with deterministic synthetic embeddings."""

import csv
import json

import pytest

from cnn_trainer.blocks import visual_split as visual
from cnn_trainer.dataset import Annotation, Dataset, ImageRecord
from cnn_trainer.errors import BlockError


def test_visual_split_merges_related_groups_and_reports_coverage(tmp_path, ctx, monkeypatch):
    np = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    pytest.importorskip("matplotlib")
    paths = [f"session{i // 2}_frame{i}.jpg" for i in range(12)]
    dataset = Dataset(
        images=[
            ImageRecord(
                path=tmp_path / path,
                relative_path=(tmp_path / path).relative_to(tmp_path),
                width=32,
                height=32,
                annotations=[Annotation(0, 1, 1, 10, 10)] if i % 2 else [],
            )
            for i, path in enumerate(paths)
        ],
        classes=["HDD"],
        source_format="test",
        root=tmp_path,
        metadata={"redundancy": {"duplicate_groups": [{"members": [paths[1], paths[2]]}]}},
    )
    manifest = tmp_path / "sessions.json"
    manifest.write_text(json.dumps({paths[3]: "same", paths[4]: "same"}))
    monkeypatch.setattr(
        visual,
        "_embed",
        lambda records, *_: np.array(
            [[index % 2 * 10, index / 100] for index in range(len(records))]
        ),
    )

    visual.visual_split(
        dataset,
        ctx,
        group_regex=r"^(session\d+)_",
        group_manifest=str(manifest),
        max_clusters=3,
        min_cluster_size=2,
    )

    assert len({record.split for record in dataset.images[:6]}) == 1
    assert {record.split for record in dataset.images} == {"train", "val", "test"}
    report = json.loads((ctx.workdir / "visual-split.json").read_text())
    assert report["groups"] == 4
    assert report["chosen_clusters"] == 2
    plot = ctx.workdir / report["k_scores_plot"]
    assert plot.name == "visual-split-k-scores.png"
    assert plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert sum(report["images_by_split"].values()) == len(paths)
    assert all(
        "class:0" in counts and "background" in counts
        for counts in report["coverage_by_split"].values()
    )
    with (ctx.workdir / "visual-split.csv").open(newline="") as stream:
        assignments = {row["relative_path"]: row["split"] for row in csv.DictReader(stream)}
    assert assignments == {
        record.relative_path.as_posix(): record.split for record in dataset.images
    }
    with pytest.raises(BlockError, match="already has split"):
        visual.visual_split(dataset, ctx)


def test_review_compares_k_and_writes_sample_images(tmp_path, ctx, monkeypatch):
    np = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    pytest.importorskip("matplotlib")
    from PIL import Image

    records = []
    for index in range(12):
        path = tmp_path / f"frame-{index:02}.jpg"
        Image.new("RGB", (32, 24), (index * 20, 0, 0)).save(path)
        records.append(
            ImageRecord(path=path, relative_path=path.relative_to(tmp_path), width=32, height=24)
        )
    dataset = Dataset(images=records, classes=[], source_format="test", root=tmp_path)
    features = np.array([[index % 2 * 10, index / 100] for index in range(len(records))])

    def embed(records, device, batch_size, ctx, backbone):
        assert backbone == "mobilenet_v3_small"
        return features

    monkeypatch.setattr(visual, "_embed", embed)
    visual.visual_split(
        dataset,
        ctx,
        backbone="mobilenet_v3_small",
        review_only=True,
        max_clusters=3,
        min_cluster_size=2,
        samples_per_cluster=2,
    )
    assert all(record.split is None for record in records)
    assert not (ctx.workdir / "visual-split.csv").exists()
    [directory] = ctx.workdir.glob("visual-review-*")
    report = json.loads((directory / "review.json").read_text())
    assert report["backbone"] == "mobilenet_v3_small"
    assert report["recommended_k"] == 2
    assert [candidate["k"] for candidate in report["candidates"]] == [2, 3]
    assert (directory / "k-scores.png").is_file()
    for candidate in report["candidates"]:
        k = candidate["k"]
        assert sum(candidate["preview_images_by_split"].values()) == len(records)
        folder = directory / f"k-{k:02d}"
        assert (folder / "clusters.png").is_file()
        assert (folder / "samples-01.png").is_file()
        assert len(list((folder / "images").glob("*.jpg"))) == k * 2
    visual.visual_split(dataset, ctx, backbone="mobilenet_v3_small", clusters=3)
    assert {record.split for record in records} == {"train", "val", "test"}
    report = json.loads((ctx.workdir / "visual-split.json").read_text())
    assert report["chosen_clusters"] == 3
