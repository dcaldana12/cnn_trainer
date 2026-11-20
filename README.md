# cnn-trainer

[![CI](https://github.com/dcaldana12/cnn_trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/dcaldana12/cnn_trainer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

YAML-driven pipelines for object-detection datasets and YOLO training.

Load a CVAT XML export, a COCO JSON file, or an existing YOLO directory, then
combine the blocks you need. A pipeline can stop after assessing, cleaning, or
exporting a dataset.

| Blocks | Purpose |
| --- | --- |
| `stats`, `validate`, `audit_dataset` | Summarize dataset contents, check annotations, and assess quality. |
| `find_redundant_images`, `remove_flagged_images` | Find near-duplicates and blank images and remove selected candidates. |
| `create_missing_labels`, `remap_classes`, `remove_class` | Create empty labels, remap classes, or remove annotations by class. |
| `merge_datasets` | Combine same-format datasets and remap labels into a shared class list. |
| `segmentation_to_boxes` | Convert YOLO polygons to axis-aligned detection boxes. |
| `grayscale` | Convert images to grayscale stored in three RGB channels. |
| `split`, `visual_split` | Assign train, validation, and test splits. `visual_split` balances visual clusters and keeps capture groups together. |
| `export_yolo`, `write_image_list`, `make_batches` | Export YOLO datasets, image lists, or batches with optional ZIP archives. |
| `train` | Train a detector with Ultralytics YOLO. |

See the [block reference](docs/blocks.md) for parameters and aliases.

## Features

- **Composable blocks** — every step is a named block in one YAML file.
- **Dataset-first** — clean, audit, split, and export without installing PyTorch.
- **Real formats** — native CVAT for images 1.1, COCO detection JSON, and YOLO.
- **Quality tools** — class/split audits, near-duplicate and blank-image detection,
  and guarded removal.
- **Optional training** — hand the finished dataset to Ultralytics YOLO.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)

## Install

```bash
git clone https://github.com/dcaldana12/cnn_trainer.git
cd cnn_trainer

# Create a project-local virtual environment (.venv)
uv venv

# Core pipeline (PyYAML only)
uv sync

# Add dataset quality analysis (Pillow)
uv sync --extra images

# Add Ultralytics training
uv sync --extra train

# Everything, including development tools
uv sync --all-extras
```

`uv venv` keeps every dependency inside `./.venv`. `uv run` automatically uses that environment, so activation is
optional:

```bash
source .venv/bin/activate   # optional; not required for uv run
uv run cnn-trainer blocks -v
```

To use the CLI outside the repository, install it into uv's isolated tool
environment:

```bash
uv tool install "cnn-trainer[images] @ git+https://github.com/dcaldana12/cnn_trainer.git"
```

## Quick start

```bash
# List every block and its parameters
uv run cnn-trainer blocks -v

# Load the dataset and print the plan without executing it
uv run cnn-trainer run examples/pipeline.yaml --dry-run

# Run the pipeline
uv run cnn-trainer run examples/pipeline.yaml
```

The example pipelines use `/data/...` paths. Edit the `source` section for your
dataset first.

## A minimal pipeline

```yaml
run:
  name: demo
  seed: 42

source:
  format: coco
  root: /data/my-dataset
  annotations: annotations/instances_default.json
  images: images

pipeline:
  - block: validate
  - block: audit_dataset
  - block: split
    ratios: {train: 0.7, val: 0.2, test: 0.1}
  - block: export_yolo
    path: build/yolo
    overwrite: true
```

Each step is one block; all other keys are that block's parameters. Unknown
blocks and unknown parameters are rejected before anything runs.

## Documentation

| Page | Contents |
| --- | --- |
| [Pipeline format](docs/pipeline.md) | YAML schema, path resolution, run outputs, examples |
| [Source formats](docs/sources.md) | CVAT, COCO, YOLO, and label conventions |
| [Blocks](docs/blocks.md) | Full parameter reference for every block |
| [Dataset quality](docs/quality.md) | Audits, duplicate detection, removal workflow |
| [Development](docs/development.md) | uv setup, tests, adding blocks, releases |

## Project layout

```text
src/cnn_trainer/   Package: adapters, blocks, registry, CLI
tests/             Synthetic datasets and pipeline tests
examples/          Runnable YAML configurations
docs/              Extended documentation
```

## Contributing

Issues and pull requests are welcome. Read
[CONTRIBUTING.md](CONTRIBUTING.md) for the local setup, test commands, and block
guidelines. By participating you agree to the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Security

Please report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE)
