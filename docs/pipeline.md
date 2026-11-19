# Pipeline format

A pipeline is a single YAML file. It has three top-level sections: `run`,
`source`, and `pipeline`.

```yaml
run:
  name: my-run                 # used for the default work directory
  seed: 42                     # default seed for split and training
  workdir: runs/my-run         # optional; relative to this YAML file

source:
  format: cvat_xml             # cvat_xml | coco | yolo
  root: /data/my-dataset       # relative to this YAML file, or absolute
  annotations: annotations.xml # relative to root
  images: images               # relative to root

pipeline:
  - block: validate
    num_classes: 2

  - block: split
    ratios:
      train: 0.7
      val: 0.2
      test: 0.1

  - block: export_yolo
    path: build/yolo           # relative to this YAML file
```

## Top-level keys

| Key | Required | Purpose |
| --- | --- | --- |
| `run` | no | Run name, default seed, and work directory |
| `source` | yes | Dataset format and its paths |
| `pipeline` | yes | Ordered list of blocks |

Unknown top-level keys, unknown `run` keys, unknown `source` options, unknown
blocks, unknown block parameters, and missing required parameters are all rejected
before any block runs.

## `run`

| Field | Default | Meaning |
| --- | --- | --- |
| `name` | YAML file stem | Run name |
| `seed` | `42` | Default seed used by `split` and `train` |
| `workdir` | `runs/<name>` | Directory for reports and manifests |

## `pipeline` steps

Each step needs a `block` key. Every other key is passed to that block.

```yaml
pipeline:
  - block: split
    ratios: {train: 0.8, val: 0.2}
    seed: 7
  - block: export_yolo
```

A step can be kept in the file but skipped:

```yaml
- block: train
  enabled: false
  model: yolov8n.pt
```

Aliases are accepted anywhere a block name is used, so
`block: check_dataset` runs `validate`.

## Path resolution

Paths in the YAML are resolved consistently:

| Field | Resolved against |
| --- | --- |
| `run.workdir` | the YAML file's directory |
| `export_yolo.path` | the YAML file's directory |
| `write_image_list.path` | the YAML file's directory |
| `make_batches.path` | the YAML file's directory |
| `train.data`, `train.project`, `train.export_dir` | the YAML file's directory |
| `source.root` | the YAML file's directory |
| `source.annotations`, `source.images`, `source.labels`, `source.data_yaml` | `source.root` |

Absolute paths and `~` are supported everywhere.

## The in-memory dataset

The adapters normalize every format to the same representation:

- image path and path relative to the dataset root;
- image width and height;
- class names;
- annotations as absolute pixel `x1, y1, x2, y2`;
- an optional `train` / `val` / `test` split tag;
- `has_label`, which marks images whose label file is missing.

Image pixels are not decoded while loading. They are read only by the image
analysis blocks and when an export or training block copies files.

## Run outputs

A successful run writes these files to `run.workdir`:

| File | Written by |
| --- | --- |
| `run.json` | Pipeline runner |
| `dataset.json` | Pipeline runner |
| `validation.json` | `validate` |
| `stats.json` | `stats` |
| `audit.json`, `audit.md` | `audit_dataset` |
| `redundancy.json`, `candidates.txt` | `find_redundant_images` |
| `removed.json` | `remove_flagged_images` |
| `training_dataset/`, `training/` | `train` |

`run.json` contains the resolved configuration, each block's status and duration,
the final dataset statistics, and the run status. If a block fails, `run.json` is
still written with `"status": "failed"` and the failing block.

## Dry runs

```bash
uv run cnn-trainer run pipeline.yaml --dry-run
```

A dry run loads the source dataset, validates block names and parameters, and
prints the plan. It does not create the work directory and does not execute any
block.

## Examples

### Dataset-only preparation

```yaml
run:
  name: clean-cvat
  seed: 42

source:
  format: cvat_xml
  root: /data/cvat/HDD_Detect_2
  annotations: annotations.xml
  images: images

pipeline:
  - block: validate
    num_classes: 9
  - block: audit_dataset
  - block: create_missing_labels
  - block: remap_classes
    mapping:
      4: 0
      7: null
    prune_unused: true
  # Review runs/clean-cvat/redundancy.json before removing anything.
  - block: find_redundant_images
    similarity_threshold: 6
  - block: remove_flagged_images
    max_fraction: 0.3
  - block: split
    ratios: {train: 0.7, val: 0.2, test: 0.1}
  - block: export_yolo
    path: build/hdd-yolo
    overwrite: true
```

### COCO to YOLO, then train

```yaml
source:
  format: coco
  root: /data/coco-export
  annotations: annotations/instances_default.json
  images: images

pipeline:
  - block: validate
  - block: split
  - block: export_yolo
    path: build/coco-yolo
    overwrite: true
  - block: train
    model: yolov8n.pt
    epochs: 100
```

### Quality pass only

See [`examples/pipeline_quality.yaml`](../examples/pipeline_quality.yaml):

```yaml
pipeline:
  - block: audit_dataset
    min_instances_per_class: 50
  - block: find_redundant_images
    similarity_threshold: 6
  - block: remove_flagged_images
    max_fraction: 0.25
    dry_run: true
  - block: audit_dataset
```
