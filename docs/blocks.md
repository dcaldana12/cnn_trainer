# Blocks

Blocks are plain Python functions registered under a name and optional aliases:

```python
def my_block(dataset: Dataset, ctx: Context, *, param=default) -> Dataset:
    ...
    return dataset
```

They receive the dataset and a run context, and return the dataset. Use the
registry to check which parameters exist:

```bash
uv run cnn-trainer blocks -v
```

Unknown block names and unknown parameters are configuration errors, reported
before execution.

## Overview

| Block | Aliases | Purpose |
| --- | --- | --- |
| `stats` | `summary` | Counts and split statistics |
| `validate` | `check_dataset` | Structural and geometry checks |
| `audit_dataset` | `analyze`, `dataset_audit`, `insights` | Quality report and tips |
| `create_missing_labels` | `create_empty_label_file` | Treat missing labels as background |
| `remap_classes` | `change_label`, `change_labels` | Merge, rename, or drop classes |
| `remove_class` | `remove_label` | Remove annotations of selected classes |
| `find_redundant_images` | `find_duplicates`, `find_uninformative` | Flag duplicates and blank images |
| `remove_flagged_images` | `remove_redundant`, `drop_candidates`, `remove_images` | Drop flagged images |
| `split` | | Assign train/val/test |
| `export_yolo` | | Write a YOLO/Ultralytics dataset |
| `write_image_list` | `write_path`, `write_train_txt` | Write an image path list |
| `make_batches` | `split_batches` | Export fixed-size batch archives |
| `train` | | Train with Ultralytics |

## `stats` (alias: `summary`)

Print dataset counts and write `stats.json` to the run directory. No parameters.

## `validate` (alias: `check_dataset`)

Check image/label pairing, class ids, box geometry, image bounds, duplicate paths,
label-stem collisions, and orphan label files.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `num_classes` | `null` | Maximum class id + 1; pads missing class names |
| `strict` | `true` | Raise when errors are found |
| `check_bounds` | `true` | Require boxes to stay inside the image |
| `require_labels` | `true` | Treat missing label files as errors |
| `orphan_labels` | `warning` | `ignore`, `warning`, or `error` |
| `write_report` | `true` | Write `validation.json` |
| `max_logged` | `20` | Number of messages printed to the console |

```yaml
- block: validate
  num_classes: 9
  require_labels: false   # allow missing labels for now
  orphan_labels: error
```

`strict: false` reports findings and lets the pipeline continue. It is useful
while inspecting a new export, but not for production runs.

## `audit_dataset` (aliases: `analyze`, `dataset_audit`, `insights`)

Read-only quality report to run before training. It writes `audit.json` and a
human-readable `audit.md`, records the report in `dataset.metadata["audit"]`, and
prints the most important findings.

It checks:

- class instance counts, images per class, and imbalance ratio;
- per-split class coverage (a class missing from `val` or `test`);
- empty images, missing label files, and invalid class ids;
- tiny boxes, boxes touching image borders, small images, extreme aspect ratios;
- validation/test size and unassigned images;
- redundancy candidates and cross-split duplicate groups from
  `find_redundant_images`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `min_instances_per_class` | `20` | Warn below this many instances |
| `min_images_per_class` | `10` | Warn below this many images per class |
| `max_imbalance_ratio` | `10.0` | Warn above this largest/smallest ratio |
| `tiny_box_area` | `0.0005` | Relative box area considered tiny |
| `max_tiny_share` | `0.1` | Tiny-box share that becomes a warning |
| `border_margin` | `0.01` | Relative margin that counts as touching the border |
| `min_image_side` | `64` | Warn about images smaller than this |
| `max_aspect_ratio` | `4.0` | Warn above this width/height ratio |
| `min_val_images` | `20` | Warn when validation has fewer images |
| `write_report` | `true` | Write `audit.json` |
| `write_markdown` | `true` | Write `audit.md` |
| `max_logged` | `12` | Findings printed to the console |

Findings have two levels: `warning` (likely to hurt training) and `tip`
(improvement worth considering). See [Dataset quality](quality.md).

## `create_missing_labels` (alias: `create_empty_label_file`)

Marks images without label files as background images. Empty `.txt` files are
written during export. No parameters.

Only use this when a missing label really means "no objects". Otherwise fix the
source export first.

## `remap_classes` (aliases: `change_label`, `change_labels`)

Merge, rename, or drop classes using old → new mappings.

```yaml
- block: remap_classes
  mapping:
    4: 0          # merge old class 4 into class 0
    7: null       # drop class 7 annotations
    car: vehicle  # rename class "car" in place
  prune_unused: true
```

Keys and values may be class ids or names, including quoted ids such as `"4"`.
A `null` value drops annotations. A new name renames the source class; an existing
name merges into that class.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `mapping` | required | Old class → new class, or `null` to drop |
| `prune_unused` | `false` | Remove classes with no remaining annotations |

Mappings are applied in the order needed to resolve names consistently. Use
`prune_unused: true` after drops so the exported `data.yaml` does not keep unused
classes.

## `remove_class` (alias: `remove_label`)

Remove every annotation of one or more classes. Accepts ids or names.

```yaml
- block: remove_class
  classes: [7, "person"]
  prune_unused: true
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `classes` | required | Class ids or names to remove |
| `prune_unused` | `false` | Remove classes left with no annotations |

## `find_redundant_images` (aliases: `find_duplicates`, `find_uninformative`)

Perceptual-hash pass over the images. It groups near-duplicates and keeps one
representative per group, and marks flat/low-contrast images without boxes as
`blank`. Nothing is deleted: candidates are stored in
`dataset.metadata["image_flags"]["redundant"]` and listed in `redundancy.json`
and `candidates.txt`. Duplicate groups that span splits are marked `cross_split`.

This block needs Pillow (`uv sync --extra images`) and reads image pixels, so it
is the slowest preprocessing block on large datasets.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `flag` | `redundant` | Metadata key for a following `remove_flagged_images` |
| `hash_size` | `8` | dHash size; 8 is 64 bits and works well for photos |
| `similarity_threshold` | `6` | Maximum Hamming distance (0–7) for duplicates |
| `min_contrast` | `5.0` | Grayscale std-dev below which an unlabeled image is blank |
| `protect_splits` | `null` | Splits whose images must never be flagged |
| `respect_splits` | `false` | Only compare images inside the same split |
| `write_report` | `true` | Write `redundancy.json` |
| `write_list` | `true` | Write `candidates.txt` with one path per line |
| `max_logged` | `10` | Candidates printed to the console |

When a group has several images, the keeper is chosen by: protected split first,
then the number of boxes, rare-class coverage, `test`/`val`/`train` priority, and
image area. Run this **before** `split` so duplicates cannot leak across train and
validation. If you run it after `split`, consider `protect_splits: [test]` and
`respect_splits: true`.

`redundancy.json` contains:

- `duplicate_groups`: keeper, members, size, and `cross_split` for each group;
- `candidates`: path, split, reason, and similarity or contrast;
- `counts`: number of duplicate and blank candidates;
- `unreadable`: images Pillow could not decode.

See [Dataset quality](quality.md) for tuning advice.

## `remove_flagged_images` (aliases: `remove_redundant`, `drop_candidates`, `remove_images`)

Drop images flagged by `find_redundant_images`. Only the in-memory dataset changes;
source files stay untouched. Downstream `export_yolo` and `train` blocks no longer
see those images or their labels.

```yaml
- block: remove_flagged_images
  flag: redundant
  reasons: [duplicate, blank]
  max_fraction: 0.3
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `flag` | `redundant` | Metadata key written by `find_redundant_images` |
| `paths` | `null` | Explicit relative paths instead of flagged images |
| `reasons` | `null` | Only remove `duplicate`, `blank`, or `explicit` candidates |
| `protect_splits` | `null` | Never remove images in these splits |
| `max_fraction` | `0.5` | Refuse to remove more than this share of the dataset |
| `force` | `false` | Allow removals above `max_fraction` |
| `dry_run` | `false` | Report what would be removed without changing the dataset |
| `write_report` | `true` | Write `removed.json` |
| `max_logged` | `10` | Removed paths printed to the console |

The `max_fraction` guard exists because an aggressive `similarity_threshold` can
classify most of a dataset as duplicates. Start with `dry_run: true`, inspect
`redundancy.json`, then remove.

`candidates.txt` holds one relative path per line. After reviewing it, pass the
edited list back with `paths:` to remove exactly those images without rerunning
the detector.

## `split`

Assign images to `train`, `val`, and `test`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `ratios` | `{train: 0.7, val: 0.2, test: 0.1}` | Fractions; normalized automatically |
| `seed` | run seed | Random seed |
| `stratify` | `true` | Keep each dominant class distributed across splits |
| `shuffle` | `true` | Shuffle before assigning |
| `overwrite` | `true` | Replace existing split assignments |

When `stratify` is enabled, images are grouped by their dominant class (or
`background`) and each group is split proportionally.

The allocation is deterministic for a given seed and input order. Use
`overwrite: false` to protect splits that were assigned earlier, for example by a
group-aware preprocessing step.

## `export_yolo`

Write a complete YOLO/Ultralytics dataset:

```text
build/yolo/
  data.yaml
  images/train/...
  images/val/...
  images/test/...
  labels/train/...
  labels/val/...
  labels/test/...
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `path` | `workdir/dataset` | Output directory |
| `default_split` | `train` | Split for images without an assignment |
| `mode` | `copy` | `copy`, `symlink`, or `hardlink` |
| `overwrite` | `false` | Replace a non-empty output directory |
| `clamp` | `false` | Clamp out-of-bounds boxes instead of failing |
| `absolute_path` | `false` | Write an absolute `path` into `data.yaml` |

`data.yaml` contains `nc`, `names`, and every split that has images. By default it
does **not** contain `path`, so Ultralytics uses the YAML file's parent directory
and the exported dataset (or a batch ZIP) can be moved safely. Set
`absolute_path: true` if you need to pin the dataset root.

Exports refuse to write into a non-empty directory unless `overwrite: true` is
set. Empty label files are written for background images.

Use `clamp: true` only for known review-able overflow; otherwise fix boxes in the
source or with `validate`.

## `write_image_list` (aliases: `write_path`, `write_train_txt`)

Write one image path per line, optionally filtered by split.

```yaml
- block: write_image_list
  path: build/train.txt
  split: train
  prefix: data/images/train
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `path` | required | Output text file |
| `split` | `null` | Only include this split |
| `prefix` | `null` | Path prefix added to each relative image path |
| `absolute` | `false` | Write absolute image paths |
| `default_split` | `train` | Split assumed when an image has none |

## `make_batches` (alias: `split_batches`)

Export fixed-size YOLO batches, each with its own `data.yaml`, and optionally zip
each batch.

```yaml
- block: make_batches
  path: build/batches
  batch_size: 100
  split: train
  zip_archives: true
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `path` | required | Output directory |
| `batch_size` | `100` | Maximum images per batch |
| `split` | `null` | Only include this split |
| `default_split` | `train` | Split assumed when an image has none |
| `zip_archives` | `true` | Create `batch_001.zip`, ... |
| `overwrite` | `false` | Replace a non-empty output directory |

Each batch is a complete portable YOLO dataset. Zip archives are useful for
uploading one batch at a time to a training machine.

## `train`

Train with Ultralytics YOLO. If `data` is omitted, the current dataset is exported
to `workdir/training_dataset` first, so a separate `export_yolo` block is not
required. Ultralytics requires both a `train` and a `val` split; add a `split`
block before `train` (or pass an existing `data.yaml` with `data`).

```yaml
- block: train
  backend: ultralytics
  model: yolov8n.pt
  epochs: 100
  imgsz: 640
  batch: 16
  device: 0
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `backend` | `ultralytics` | Training backend |
| `model` | `yolov8n.pt` | Ultralytics model name or checkpoint path |
| `epochs` | `100` | Training epochs |
| `imgsz` | `640` | Input image size |
| `batch` | `16` | Batch size |
| `device` | `null` | Device id, list, `cpu`, or `mps` |
| `workers` | `8` | Dataloader workers |
| `project` | `workdir/training` | Ultralytics project directory |
| `name` | `train` | Run name inside the project |
| `exist_ok` | `true` | Allow an existing run directory |
| `seed` | run seed | Training seed |
| `data` | `null` | Use an existing `data.yaml` instead of exporting |
| `export_dir` | `workdir/training_dataset` | Internal export directory |
| `extra` | `{}` | Additional Ultralytics `train()` arguments |

Any Ultralytics argument not listed above can be passed through `extra`:

```yaml
- block: train
  extra:
    patience: 20
    plots: true
    resume: false
```

Training requires the `train` extra:

```bash
uv sync --extra train
uv run cnn-trainer run pipeline.yaml
```
