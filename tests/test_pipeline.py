"""End-to-end pipeline and CLI tests."""

from __future__ import annotations

import json
import random
import textwrap

import pytest
import yaml

from cnn_trainer.cli import main
from cnn_trainer.errors import BlockError, ConfigError
from cnn_trainer.pipeline import run_pipeline


def _write_config(path, text):
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
    return path


def _config_text(*, workdir="runs/test-run", pipeline=None):
    if pipeline is None:
        pipeline = """\
  - block: validate
    num_classes: 2
  - block: split
    ratios:
      train: 0.5
      val: 0.25
      test: 0.25
    seed: 7
    stratify: false
  - block: export_yolo
    path: build/yolo
"""
    return f"""\
run:
  name: test-run
  seed: 7
  workdir: {workdir}

source:
  format: coco
  root: coco
  annotations: annotations/instances_default.json
  images: images

pipeline:
{pipeline}"""


def test_pipeline_end_to_end(coco_root, tmp_path):
    config = _write_config(
        tmp_path / "pipeline.yaml",
        _config_text(
            pipeline="""\
  - block: validate
    num_classes: 2
  - block: create_missing_labels
  - block: remap_classes
    mapping:
      1: 0
    prune_unused: true
  - block: split
    ratios:
      train: 0.5
      val: 0.25
      test: 0.25
    seed: 7
    stratify: false
  - block: export_yolo
    path: build/yolo
"""
        ),
    )

    manifest = run_pipeline(config)

    assert manifest["status"] == "success"
    assert manifest["classes"] == ["cat"]
    assert manifest["dataset"]["images"] == 4
    assert [block["status"] for block in manifest["blocks"]] == ["ok"] * 5

    run_dir = tmp_path / "runs" / "test-run"
    assert (run_dir / "run.json").is_file()
    assert (run_dir / "dataset.json").is_file()
    saved = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert saved["status"] == "success"

    data_path = tmp_path / "build" / "yolo" / "data.yaml"
    assert data_path.is_file()
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    assert data["names"] == {0: "cat"}
    assert data["nc"] == 1


def test_pipeline_dry_run_does_not_create_workdir(coco_root, tmp_path):
    config = _write_config(
        tmp_path / "pipeline.yaml",
        _config_text(workdir="runs/dry"),
    )

    manifest = run_pipeline(config, dry_run=True)

    assert manifest["status"] == "dry-run"
    assert manifest["dataset"]["images"] == 4
    assert len(manifest["blocks"]) == 3
    assert not (tmp_path / "runs" / "dry").exists()


def test_pipeline_rejects_unknown_block(coco_root, tmp_path):
    config = _write_config(
        tmp_path / "pipeline.yaml",
        _config_text(pipeline="  - block: does_not_exist\n"),
    )

    with pytest.raises(ConfigError, match="Unknown block"):
        run_pipeline(config)


def test_pipeline_rejects_unknown_parameter(coco_root, tmp_path):
    config = _write_config(
        tmp_path / "pipeline.yaml",
        _config_text(
            pipeline="""\
  - block: validate
    bogus_parameter: 3
"""
        ),
    )

    with pytest.raises(ConfigError, match="does not accept parameter"):
        run_pipeline(config)


def test_pipeline_rejects_unknown_source_option(coco_root, tmp_path):
    config = _write_config(
        tmp_path / "pipeline.yaml",
        """\
source:
  format: coco
  root: coco
  image: images
pipeline:
  - block: stats
""",
    )

    with pytest.raises(ConfigError, match="Unknown source option"):
        run_pipeline(config)


def test_pipeline_rejects_unknown_run_option(coco_root, tmp_path):
    config = _write_config(
        tmp_path / "pipeline.yaml",
        """\
run:
  seeds: 7
source:
  format: coco
  root: coco
  annotations: annotations/instances_default.json
  images: images
pipeline:
  - block: stats
""",
    )

    with pytest.raises(ConfigError, match="Unknown run option"):
        run_pipeline(config)


def test_pipeline_supports_legacy_aliases(coco_root, tmp_path):
    config = _write_config(
        tmp_path / "pipeline.yaml",
        _config_text(
            pipeline="""\
  - block: check_dataset
    num_classes: 2
  - block: change_labels
    mapping:
      1: 0
  - block: summary
"""
        ),
    )

    manifest = run_pipeline(config)

    assert [block["canonical"] for block in manifest["blocks"]] == [
        "validate",
        "remap_classes",
        "stats",
    ]


def test_pipeline_failure_writes_manifest(yolo_root, tmp_path):
    config = _write_config(
        tmp_path / "pipeline.yaml",
        """\
run:
  name: fail-run
  workdir: runs/fail
source:
  format: yolo
  root: yolo
pipeline:
  - block: validate
""",
    )

    with pytest.raises(BlockError, match="validation failed"):
        run_pipeline(config)

    run_file = tmp_path / "runs" / "fail" / "run.json"
    assert run_file.is_file()
    saved = json.loads(run_file.read_text(encoding="utf-8"))
    assert saved["status"] == "failed"
    assert saved["blocks"][0]["status"] == "failed"


def test_pipeline_audits_and_removes_duplicates(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    root = tmp_path / "coco"
    images = root / "images"
    images.mkdir(parents=True)

    def write_noise(name, seed):
        rng = random.Random(seed)
        data = bytes(rng.randrange(256) for _ in range(64 * 64))
        Image.frombytes("L", (64, 64), data).save(images / name)

    entries = [("a1.png", 1), ("a2.png", 1), ("b1.png", 2), ("b2.png", 2)]
    for name, seed in entries:
        write_noise(name, seed)

    payload = {
        "images": [
            {"id": index + 1, "file_name": name, "width": 64, "height": 64}
            for index, (name, _) in enumerate(entries)
        ],
        "categories": [{"id": 1, "name": "object"}],
        "annotations": [
            {
                "id": index + 1,
                "image_id": index + 1,
                "category_id": 1,
                "bbox": [10, 10, 20, 20],
                "iscrowd": 0,
            }
            for index, _ in enumerate(entries)
        ],
    }
    annotations_dir = root / "annotations"
    annotations_dir.mkdir()
    (annotations_dir / "instances_default.json").write_text(json.dumps(payload), encoding="utf-8")

    config = _write_config(
        tmp_path / "pipeline.yaml",
        _config_text(
            pipeline="""\
  - block: audit_dataset
    min_instances_per_class: 1
  - block: find_redundant_images
    min_contrast: 0.0
    similarity_threshold: 0
  - block: remove_flagged_images
    max_fraction: 0.9
""",
        ),
    )

    manifest = run_pipeline(config)

    assert manifest["status"] == "success"
    assert [block["status"] for block in manifest["blocks"]] == ["ok"] * 3
    assert manifest["dataset"]["images"] == 2
    run_dir = tmp_path / "runs" / "test-run"
    assert (run_dir / "audit.json").is_file()
    assert (run_dir / "audit.md").is_file()
    assert (run_dir / "redundancy.json").is_file()
    assert (run_dir / "candidates.txt").is_file()
    assert (run_dir / "removed.json").is_file()

    removed = json.loads((run_dir / "removed.json").read_text(encoding="utf-8"))
    assert removed["removed"] == 2


def test_cli_blocks(capsys):
    assert main(["blocks"]) == 0
    output = capsys.readouterr().out
    assert "export_yolo" in output
    assert "aliases:" in output


def test_cli_dry_run(coco_root, tmp_path, capsys):
    config = _write_config(
        tmp_path / "pipeline.yaml",
        _config_text(workdir="runs/cli"),
    )

    assert main(["run", str(config), "--dry-run"]) == 0
    captured = capsys.readouterr()
    assert "Dry run OK" in captured.out
