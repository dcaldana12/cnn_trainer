"""Verify same-format merges through the pipeline and YOLO exporter."""

from __future__ import annotations

import shutil

import pytest
import yaml

from cnn_trainer.pipeline import run_pipeline


@pytest.mark.parametrize(
    ("format_name", "fixture", "annotation_file", "old", "new", "classes", "label_ids"),
    [
        ("yolo", "yolo_root", "data.yaml", "cat", "bird", ["dog", "cat", "bird"], ([1], [2])),
        (
            "coco",
            "coco_root",
            "annotations/instances_default.json",
            '"cat"',
            '"bird"',
            ["dog", "cat", "bird"],
            ([1, 0], [2, 0]),
        ),
        (
            "cvat_xml",
            "cvat_root",
            "annotations.xml",
            "car",
            "bird",
            ["person", "car", "bird"],
            ([0, 1], [0, 2]),
        ),
    ],
)
def test_merge_preserves_images_and_reindexes_annotations(
    tmp_path, request, format_name, fixture, annotation_file, old, new, classes, label_ids
):
    first = request.getfixturevalue(fixture)
    second = tmp_path / "second"
    shutil.copytree(first, second)
    target = second / annotation_file
    target.write_text(target.read_text().replace(old, new))
    original = (first / annotation_file).read_bytes()

    source = {"format": format_name, "root": first.name}
    additional = "second"
    if format_name == "coco":
        source["images"] = "images"
        additional = {"root": "second", "images": "images"}
    config = {
        "source": source,
        "pipeline": [
            {"block": "merge_datasets", "paths": [additional]},
            {"block": "export_yolo", "path": "merged"},
        ],
    }
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text(yaml.safe_dump(config))

    assert run_pipeline(pipeline)["status"] == "success"
    export = tmp_path / "merged"
    data = yaml.safe_load((export / "data.yaml").read_text())
    assert list(data["names"].values()) == classes
    for index, expected in enumerate(label_ids, start=1):
        relative = f"train/dataset-{index:02d}/a"
        assert (export / "images" / f"{relative}.png").is_file()
        labels = (export / "labels" / f"{relative}.txt").read_text().splitlines()
        assert [int(line.split()[0]) for line in labels] == expected
    assert (first / annotation_file).read_bytes() == original
    if format_name == "yolo":
        assert (export / "images/val/dataset-02/b.png").is_file()
        assert (export / "labels/val/dataset-02/b.txt").read_text().split()[0] == "0"
