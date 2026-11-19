# Dataset quality

A clean split matters more than another training epoch. `cnn-trainer` provides
three blocks for the quality stage:

| Block | Changes the dataset? | Main output |
| --- | --- | --- |
| `audit_dataset` | No | `audit.json`, `audit.md` |
| `find_redundant_images` | No | `redundancy.json`, `candidates.txt` |
| `remove_flagged_images` | Yes, in memory | `removed.json` |

Run the whole stage before `split` and before `train`.

```yaml
pipeline:
  - block: audit_dataset
  - block: create_missing_labels
  - block: remap_classes
    mapping: {4: 0, 7: null}
    prune_unused: true

  - block: find_redundant_images
    similarity_threshold: 6
    min_contrast: 5.0

  # Review the report, then switch dry_run to false.
  - block: remove_flagged_images
    max_fraction: 0.3
    dry_run: true

  - block: audit_dataset
  - block: split
```

Requirements:

```bash
uv sync --extra images
```

## Auditing the dataset

`audit_dataset` is a read-only pass over the in-memory dataset. It does not
decode images, so it is fast even for large datasets. It reports:

- class instance counts and imbalance;
- images per class and classes with no annotations;
- class coverage in `train`, `val`, and `test`;
- empty images and missing label files;
- tiny boxes, border-touching boxes, and invalid class ids;
- very small images and extreme aspect ratios;
- split sizes and unassigned images;
- redundancy candidates and cross-split duplicate groups.

Reports are written to `run.workdir/audit.json` and `audit.md`. Findings are
classified as:

- **warning** — likely to hurt training or evaluation, for example no `val` split,
  a class absent from validation, or hundreds of missing labels;
- **tip** — worth reviewing, for example few images per class or objects near the
  image border.

Tune the thresholds if your dataset has different expectations:

```yaml
- block: audit_dataset
  min_instances_per_class: 50
  min_images_per_class: 25
  min_val_images: 100
  max_imbalance_ratio: 5
```

`audit.md` is a compact review artifact for teammates. Commit or attach it to a
dataset release if you track dataset versions.

## Finding images that add little value

`find_redundant_images` computes a 64-bit dHash for every image and compares
hashes by Hamming distance. It also measures grayscale contrast to find blank
images. Two kinds of candidate are produced:

| Reason | Meaning |
| --- | --- |
| `duplicate` | Very similar to a kept image in the same group |
| `blank` | Low contrast and no annotations |

The block never removes anything. Candidates are listed in
`run.workdir/redundancy.json` and as plain paths in `candidates.txt`.

### Choosing a keeper

For each duplicate group, the block keeps the most useful image:

1. images in a protected split (`protect_splits`);
2. more annotations;
3. coverage of rare classes;
4. `test` over `val` over `train`;
5. larger image area;
6. deterministic path order as a final tie-breaker.

### Tuning similarity

- `similarity_threshold: 0` only exact hashes.
- `similarity_threshold: 4–6` catches repeated frames and near-identical shots.
- Values above 6 are rejected; above the default the risk of false positives
  grows quickly.

If you need to distinguish images that differ only in color, inspect
`duplicate_groups` manually. Hashing is grayscale by design.

### Splits and leakage

Duplicate frames that end up in different splits inflate validation metrics. The
report marks such groups with `cross_split: true`, and `audit_dataset` adds a
warning when splits already exist.

Recommended order:

1. `find_redundant_images`
2. `remove_flagged_images`
3. `split`
4. `export_yolo`

If you must run the quality stage after splitting:

```yaml
- block: find_redundant_images
  respect_splits: true      # only compare within one split
  protect_splits: [test]    # never flag test images
```

### Reviewing and removing

Start in dry-run mode:

```yaml
- block: remove_flagged_images
  dry_run: true
  max_fraction: 0.3
```

Inspect `candidates.txt`. You can delete lines you want to keep and then pass the
edited file contents back explicitly:

```yaml
- block: remove_flagged_images
  paths:
    - train/frame_0012.jpg
    - train/frame_0013.jpg
```

Removal only changes the in-memory dataset. Source files are untouched, and
downstream exports simply omit the removed images and their labels.

The `max_fraction` guard refuses to remove more than the configured share of the
dataset. Raise it deliberately with `force: true` only after reviewing the report.

## Heavier visual analysis

The optional `visual_split` block extracts pretrained image embeddings,
normalizes them, projects them with PCA, and fits K-Means. Set `backbone`
to `resnet50` (the default), `resnet18`, `efficientnet_b0`,
`mobilenet_v3_small`, or `vit_b_16`. Automatic cluster
selection prefers candidates that meet the minimum cluster size, using
silhouette and Davies–Bouldin scores to rank them. Optional UMAP coordinates
help inspect the assignments. Clustering uses the PCA features.

Add `visual_split` to your pipeline YAML and run
`uv run --extra visual cnn-trainer run path/to/pipeline.yaml` from the repository
root. The first run needs pretrained weights from torchvision
or an existing weights cache. Use `--extra umap` alongside `--extra visual`
and set `umap: true` to include UMAP coordinates.

To compare K values before splitting, use this as the final block of a review pipeline:

```yaml
  - block: visual_split
    backbone: efficientnet_b0
    review_only: true
    max_clusters: 12
    samples_per_cluster: 4
```

The run logs a `visual-review-*/` folder containing `k-scores.png`, a
`review.json` with the candidate scores and preview split coverage when groups
permit it, and one folder per K with projections, cluster sizes, sample sheets,
and individual sample thumbnails. Include the same grouping options in both
runs to make the preview comparable. To apply your choice, remove
`review_only: true`, set `clusters` to the selected K, and run the split
pipeline with the same backbone.

The splitter balances image counts, cluster coverage, class presence, and
background across the requested splits. It assigns whole groups, which can
make the achieved ratios differ from the targets. Supply a JSON group manifest
mapping dataset-relative image paths to verified capture or session IDs, or a
`group_regex` with one capture group matched against those paths. The block
also groups perceptual matches from a preceding `find_redundant_images` block
by default. Overlapping groups are merged, and unmatched images are treated
individually.

Review `visual-split.csv` for per-image assignments and projection coordinates
and `visual-split.json` for cluster scores, realized counts, and coverage.
When K is chosen automatically, `visual-split-k-scores.png` compares candidate
scores and marks the selected K and minimum cluster size. Add
`audit_dataset` and `export_yolo` after the visual split to audit and export
the assignments.

## A practical checklist

- [ ] `validate` passes with `strict: true`.
- [ ] Missing labels are intentional and fixed with `create_missing_labels`.
- [ ] Rare classes are either collected or removed/merged.
- [ ] No class is missing from `val`; `test` coverage is understood.
- [ ] Duplicate groups were reviewed and either kept or removed.
- [ ] No duplicate group spans `train` and `val`/`test`.
- [ ] Validation has enough images for a stable metric.
- [ ] Exported `data.yaml` has the expected `nc` and `names`.
