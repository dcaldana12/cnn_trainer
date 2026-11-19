# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-11-19

### Added

- YAML pipeline runner with strict configuration validation, dry runs, and
  per-block manifests.
- Source adapters for CVAT for images 1.1, COCO detection JSON, and YOLO
  directories.
- Blocks: `stats`, `validate`, `create_missing_labels`, `remap_classes`,
  `remove_class`, `split`, `export_yolo`, `write_image_list`, `make_batches`, and
  Ultralytics `train`.
- Dataset quality blocks: `audit_dataset`, `find_redundant_images`, and
  `remove_flagged_images`.
- Legacy aliases for the original standalone scripts.
- Optional extras for image analysis (`images`), training (`train`), and
  development (`dev`).
- Synthetic test suite and example pipelines.

[Unreleased]: https://github.com/dcaldana12/cnn_trainer/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dcaldana12/cnn_trainer/releases/tag/v0.1.0
