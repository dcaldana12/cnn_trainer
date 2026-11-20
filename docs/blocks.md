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
| `merge_datasets` | | Combine same-format datasets and remap labels |
| `segmentation_to_boxes` | | Convert YOLO polygons to axis-aligned boxes |
| `grayscale` | | Convert images to three-channel grayscale |
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
| `workers` | `1` | CPU threads for image decoding and fingerprinting |
| `similarity_threshold` | `6` | Maximum Hamming distance (0–7) for duplicates |
| `min_contrast` | `5.0` | Grayscale std-dev below which an unlabeled image is blank |
| `protect_splits` | `null` | Splits whose images must never be flagged |
| `respect_splits` | `false` | Only compare images inside the same split |
| `write_report` | `true` | Write `redundancy.json` |
| `write_list` | `true` | Write `candidates.txt` with one path per line |
| `review` | `false` | Write PNG contact sheets for duplicate and low-contrast candidates |
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

With `review: true`, open the `review_directory` path from `redundancy.json`
relative to the run workdir. Each `duplicates-*.png` sheet shows up to ten
candidate–keeper pairs, with candidates on the left. Low-contrast images appear
in separate `blanks-*.png` sheets with two images per row. Unreadable images are
listed with their decode errors in `redundancy.json`. Each scan writes sheets
under a fresh folder in `redundancy-review/`.

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

## `merge_datasets`

Merge the pipeline source with one or more additional datasets. All inputs use
the pipeline source format (`yolo`, `coco`, or `cvat_xml`). List roots relative
to the pipeline YAML, or use mappings with `root` and source options such as
`images`, `annotations`, or `class_names`.

```yaml
source:
  format: yolo
  root: datasets/first

pipeline:
  - block: merge_datasets
    paths:
      - datasets/second
      - root: datasets/third
        data_yaml: custom-data.yaml
  - block: validate
  - block: export_yolo
    path: build/merged-yolo
```

Class names are matched exactly. Classes that appear in more input datasets
come first, with ties following the order in which the names first appear.
Each source's annotation IDs are remapped to the new class list. For YOLO
inputs, provide class names via the dataset YAML or `class_names` so the merge
can match the intended classes.

Put this block before assessment or splitting. Existing split assignments are
preserved. Output paths use a per-source prefix so identical filenames from
different datasets can coexist. The block merges records in memory, and
`export_yolo` writes the combined images and remapped label files.

## `segmentation_to_boxes`

Convert YOLO polygon rows (`class x1 y1 x2 y2 x3 y3 ...`) to tight
axis-aligned bounding boxes. Existing YOLO detection rows and class IDs are
preserved, including when both row types appear in the same label file. The
block takes no parameters.

```yaml
pipeline:
  - block: segmentation_to_boxes
  - block: validate
  - block: export_yolo
    path: build/detection
```

The export writes five-column YOLO detection labels to the destination.
The original label files remain in the source dataset.

## `grayscale`

Convert each image to grayscale and store the same intensity in all three RGB
channels. The block needs Pillow (`uv sync --extra images`) and takes no
parameters. It writes copies in a fresh `grayscale-*/` directory in the run's
workdir. Existing image names, dimensions, annotations, and splits are
preserved. The original files remain available at their source paths.

Place the block before export or training so those steps use the converted images.

```yaml
- block: grayscale
- block: export_yolo
```

## `visual_split`

Assign whole capture groups while balancing visual clusters, classes,
and background images. See [visual dataset splitting](quality.md#heavier-visual-analysis).

| Parameter | Default | Meaning |
| --- | --- | --- |
| `ratios` | train 0.7, val 0.2, test 0.1 | Target image fractions |
| `clusters` | `null` | Fixed K or automatic selection |
| `backbone` | `resnet50` | Also accepts `resnet18`, `efficientnet_b0`, `mobilenet_v3_small`, and `vit_b_16` |
| `review_only` | `false` | Compare candidate K values and leave split assignments untouched |
| `samples_per_cluster` | `4` | Representative images per cluster in review mode |
| `max_clusters` | `20` | Largest K considered in automatic selection |
| `min_cluster_size` | `10` | Preferred minimum size when selecting K |
| `group_regex` | `null` | One capture group matched against each relative image path |
| `group_manifest` | `null` | Config-relative JSON path mapping image paths to group IDs |
| `use_duplicate_groups` | `true` | Merge perceptual groups from the current pipeline |
| `batch_size` | `32` | Images per inference batch |
| `device` | `auto` | Use CUDA when available or fall back to CPU |
| `umap` | `false` | Include UMAP in the review plots or split CSV |
| `overwrite` | `false` | Replace existing split assignments |

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
  train.txt
  val.txt
  test.txt
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

`data.yaml` contains the class ID mapping under `names`, `path: .`, and one
image-list entry per populated split, such as `train: train.txt`. Each list
contains paths like `./images/train/frame.jpg` relative to the list file.
Run Ultralytics from the dataset directory when invoking it directly. The
`train` block handles this for pipeline exports. Batch ZIPs also include their
split lists and use `path: .`.

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

Train with Ultralytics YOLO. Pass backend training arguments through `extra`.
Use `extra.cfg` to load an Ultralytics training YAML with settings such as
`model`, `epochs`, `imgsz`, and `batch`.

```yaml
- block: train
  backend: ultralytics
  extra:
    cfg: configs/yolo-training.yaml
```

Explicit top-level values take precedence over `extra`, and explicit values in
`extra` take precedence over the cfg. Omitted or `null` overrides use backend
settings. If no model is configured, the block initializes `yolov8n.pt`.
If no setting supplies `data`, the block exports the current dataset to
`workdir/training_dataset` and requires train and validation splits.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `backend` | `ultralytics` | Training backend |
| `extra` | `null` | Ultralytics arguments, e.g. `cfg` |
| `model`, `epochs`, `imgsz`, `batch`, `device`, `workers` | `null` | Optional explicit overrides |
| `project`, `name`, `exist_ok`, `seed` | `null` | Optional backend run settings |
| `data` | `null` | Use an existing `data.yaml` instead of exporting |
| `export_dir` | `null` | Override the generated training dataset directory |

Training requires the `train` extra:

```bash
uv sync --extra train
uv run cnn-trainer run pipeline.yaml
```
