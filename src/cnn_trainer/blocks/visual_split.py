"""Balance visual clusters and class coverage while assigning whole capture groups."""

from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import Counter

from ..context import Context
from ..dataset import Dataset
from ..errors import BlockError
from ..registry import block
from .split import _normalize_ratios
from .visual_review import write_k_scores, write_visual_review

_BACKBONES = {
    "resnet18": "fc",
    "resnet50": "fc",
    "efficientnet_b0": "classifier",
    "mobilenet_v3_small": "classifier",
    "vit_b_16": "heads",
}


@block("visual_split", "Split by pretrained visual clusters, classes, and capture groups.")
def visual_split(
    dataset: Dataset,
    ctx: Context,
    *,
    ratios: dict | None = None,
    clusters: int | None = None,
    backbone: str = "resnet50",
    review_only: bool = False,
    samples_per_cluster: int = 4,
    max_clusters: int = 20,
    min_cluster_size: int = 10,
    group_regex: str | None = None,
    group_manifest: str | None = None,
    use_duplicate_groups: bool = True,
    batch_size: int = 32,
    device: str = "auto",
    umap: bool = False,
    overwrite: bool = False,
) -> Dataset:
    """Assign complete groups and write a CSV manifest plus clustering diagnostics."""
    if not isinstance(backbone, str) or backbone not in _BACKBONES:
        raise BlockError(f"visual_split.backbone must be one of: {', '.join(_BACKBONES)}")
    if review_only and clusters is not None:
        raise BlockError("Omit visual_split.clusters in review mode to compare candidate values")
    if (
        not review_only
        and not overwrite
        and any(record.split is not None for record in dataset.images)
    ):
        raise BlockError("Dataset already has split assignments. Set overwrite: true to re-split.")
    for name, value, minimum in (
        ("clusters", clusters, 2),
        ("max_clusters", max_clusters, 2),
        ("min_cluster_size", min_cluster_size, 1),
        ("batch_size", batch_size, 1),
        ("samples_per_cluster", samples_per_cluster, 1),
    ):
        if name == "clusters" and value is None:
            continue
        if type(value) is not int or value < minimum:
            raise BlockError(f"visual_split.{name} must be an integer >= {minimum}")

    normalized_ratios = _normalize_ratios(ratios)
    if not all(math.isfinite(value) for value in normalized_ratios.values()):
        raise BlockError("visual_split.ratios must be finite")
    ratios = {name: ratio for name, ratio in normalized_ratios.items() if ratio > 0}
    records = sorted(dataset.images, key=lambda record: record.relative_path.as_posix())
    paths = [record.relative_path.as_posix() for record in records]
    if len(paths) < 3 or len(set(paths)) != len(paths):
        raise BlockError("visual_split requires at least three images and unique relative paths")
    if clusters is not None and clusters >= len(paths):
        raise BlockError("visual_split.clusters must be smaller than the image count")
    groups = _groups(paths, dataset, ctx, group_regex, group_manifest, use_duplicate_groups)
    if not review_only and len(groups) < len(ratios):
        raise BlockError("Too few independent groups to populate every requested split")

    features = _embed(records, device, batch_size, ctx, backbone)
    labels, projected, candidates, chosen = _cluster(
        features, clusters, max_clusters, min_cluster_size, ctx.seed
    )
    coordinates = _umap(features, ctx.seed) if umap else None
    if review_only:
        if len(groups) >= len(ratios):
            for metrics, candidate_labels in candidates:
                tags = _split_tags(records, candidate_labels)
                assignments = _assign_groups(groups, tags, ratios, ctx.seed)
                counts = {split: Counter() for split in ratios}
                image_counts = Counter()
                for group, members in groups.items():
                    split = assignments[group]
                    image_counts[split] += len(members)
                    counts[split].update(tag for index in members for tag in tags[index])
                all_tags = sorted(set().union(*tags))
                metrics["preview_images_by_split"] = dict(image_counts)
                metrics["preview_coverage_by_split"] = {
                    split: {tag: counts[split][tag] for tag in all_tags} for split in ratios
                }
        directory = write_visual_review(
            ctx,
            records,
            candidates,
            projected,
            coordinates,
            chosen,
            backbone,
            min_cluster_size,
            samples_per_cluster,
        )
        ctx.logger.info(
            "Compare candidate K values in %s, then set clusters in your split pipeline", directory
        )
        return dataset

    tags = _split_tags(records, labels)
    assignments = _assign_groups(groups, tags, ratios, ctx.seed)

    counts = {split: Counter() for split in ratios}
    image_counts = Counter()
    workdir = ctx.ensure_workdir()
    with (workdir / "visual-split.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        header = ["relative_path", "group", "cluster", "split", "pca1", "pca2"]
        writer.writerow(header + (["umap1", "umap2"] if coordinates is not None else []))
        for group, members in groups.items():
            split = assignments[group]
            for index in members:
                row = [paths[index], paths[group], labels[index], split, *projected[index, :2]]
                writer.writerow(row + (list(coordinates[index]) if coordinates is not None else []))
                counts[split].update(tags[index])
                image_counts[split] += 1

    all_tags = sorted(set().union(*tags))
    report = {
        "seed": ctx.seed,
        "backbone": backbone,
        "target_ratios": ratios,
        "groups": len(groups),
        "chosen_clusters": chosen,
        "cluster_metrics": [metrics for metrics, _ in candidates],
        "images_by_split": dict(image_counts),
        "coverage_by_split": {
            split: {tag: counts[split][tag] for tag in all_tags} for split in ratios
        },
    }
    if clusters is None:
        plot = workdir / "visual-split-k-scores.png"
        write_k_scores(plot, report["cluster_metrics"], chosen, backbone, min_cluster_size)
        report["k_scores_plot"] = plot.name
        ctx.logger.info("Wrote automatic K comparison to %s", plot)
    (workdir / "visual-split.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    for group, members in groups.items():
        for index in members:
            records[index].split = assignments[group]
    ctx.logger.info(
        "Visual split: %s across %d groups and %d clusters", image_counts, len(groups), chosen
    )
    return dataset


def _split_tags(records, labels):
    tags = []
    for record, cluster in zip(records, labels, strict=True):
        classes = {f"class:{annotation.class_id}" for annotation in record.annotations}
        tags.append({f"cluster:{cluster}", *(classes or {"background"})})
    return tags


def _groups(paths, dataset, ctx, group_regex, group_manifest, use_duplicates):
    try:
        pattern = re.compile(group_regex) if group_regex is not None else None
    except (re.error, TypeError) as error:
        raise BlockError(f"Invalid visual_split.group_regex: {error}") from error
    if pattern is not None and pattern.groups != 1:
        raise BlockError("visual_split.group_regex must have exactly one capture group")
    manifest = {}
    if group_manifest is not None:
        try:
            manifest = json.loads(ctx.resolve(group_manifest).read_text(encoding="utf-8"))
        except (OSError, ValueError, AttributeError) as error:
            raise BlockError(f"Cannot read visual split group manifest: {error}") from error
        if not isinstance(manifest, dict) or any(
            not isinstance(value, str) or not value.strip() for value in manifest.values()
        ):
            raise BlockError("Group manifest must map relative image paths to nonempty group IDs")
        unknown = manifest.keys() - set(paths)
        if unknown:
            raise BlockError(f"Group manifest contains unknown image paths, e.g. {min(unknown)!r}")

    index_by_path = {path: index for index, path in enumerate(paths)}
    parents = list(range(len(paths)))

    def root(index):
        while index != parents[index]:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def join(left, right):
        left, right = root(left), root(right)
        parents[max(left, right)] = min(left, right)

    anchors = {}
    for index, path in enumerate(paths):
        if path in manifest:
            key = ("manifest", manifest[path])
            join(index, anchors.setdefault(key, index))
        match = pattern.search(path) if pattern is not None else None
        if match and match[1]:
            key = ("regex", match[1])
            join(index, anchors.setdefault(key, index))

    if use_duplicates:
        report = dataset.metadata.get("redundancy", {})
        for group in report.get("duplicate_groups", []):
            members = [index_by_path[path] for path in group["members"] if path in index_by_path]
            for index in members[1:]:
                join(members[0], index)

    groups = {}
    for index in range(len(paths)):
        groups.setdefault(root(index), []).append(index)
    return groups


def _assign_groups(groups, tags, ratios, seed):
    totals = Counter(tag for image_tags in tags for tag in image_tags)
    totals["images"] = len(tags)
    weights = {"images": 3.0}
    observed = {split: Counter() for split in ratios}
    remaining = set(ratios)
    order = list(groups)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda group: len(groups[group]), reverse=True)
    assignments = {}
    for position, group in enumerate(order):
        contribution = Counter(tag for index in groups[group] for tag in tags[index])
        contribution["images"] = len(groups[group])

        def cost(split, contribution=contribution):
            return sum(
                weights.get(tag, 1.0)
                * count
                * (2 * (observed[split][tag] - ratios[split] * totals[tag]) + count)
                / (totals[tag] * max(ratios[split] * totals[tag], 1))
                for tag, count in sorted(contribution.items())
            )

        available = [split for split in ratios if split in remaining]
        if len(order) - position > len(remaining):
            available = list(ratios)
        split = min(available, key=cost)
        assignments[group] = split
        observed[split].update(contribution)
        remaining.discard(split)
    return assignments


def _embed(records, device, batch_size, ctx, backbone):
    try:
        import numpy as np
        import torch
        from PIL import Image, ImageOps
        from torchvision.models import get_model, get_model_weights
    except (ImportError, RuntimeError) as error:
        raise BlockError(
            "Install compatible visual dependencies with uv sync --extra visual"
        ) from error

    automatic = device == "auto"
    if automatic:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx.logger.info("Extracting %s features for %d images on %s", backbone, len(records), device)
    try:
        weights = get_model_weights(backbone).DEFAULT
        transform = weights.transforms()
        model = get_model(backbone, weights=weights)
        setattr(model, _BACKBONES[backbone], torch.nn.Identity())
        model.eval()
    except (OSError, ValueError, RuntimeError) as error:
        raise BlockError(f"Cannot initialize pretrained {backbone} on {device}: {error}") from error

    def infer(batch):
        return torch.nn.functional.normalize(model(batch.to(device))).cpu().numpy()

    def fallback(error):
        if automatic and device == "cuda":
            ctx.logger.warning("CUDA inference failed, switching to CPU: %s", error)
            return "cpu"
        raise BlockError(f"{backbone} inference failed on {device}: {error}") from error

    try:
        model.to(device)
    except (RuntimeError, ValueError) as error:
        device = fallback(error)
        model.to(device)

    batches = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            images = []
            for record in records[start : start + batch_size]:
                try:
                    with Image.open(record.path) as source:
                        images.append(transform(ImageOps.exif_transpose(source).convert("RGB")))
                except (OSError, Image.DecompressionBombError) as error:
                    raise BlockError(f"Cannot embed {record.relative_path}: {error}") from error
            batch = torch.stack(images)
            try:
                batches.append(infer(batch))
            except RuntimeError as error:
                device = fallback(error)
                model.to(device)
                batches.append(infer(batch))
    return np.concatenate(batches)


def _cluster(features, clusters, max_clusters, min_cluster_size, seed):
    try:
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.metrics import (
            calinski_harabasz_score,
            davies_bouldin_score,
            silhouette_score,
        )
        from threadpoolctl import threadpool_limits
    except ImportError as error:
        raise BlockError(
            "Install visual clustering dependencies with uv sync --extra visual"
        ) from error

    with threadpool_limits(limits=2):
        projected = PCA(
            n_components=min(100, len(features) - 1, features.shape[1]), random_state=seed
        ).fit_transform(features)
        candidates = (
            [clusters] if clusters is not None else range(2, min(max_clusters + 1, len(features)))
        )
        results = []
        for k in candidates:
            labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(projected)
            sizes = np.bincount(labels, minlength=k)
            if not sizes.min():
                continue
            # Bound silhouette work and include every cluster in the sample.
            sample = np.arange(len(labels))
            if len(sample) > 5000 and k < 5000:
                _, representatives = np.unique(labels, return_index=True)
                pool = np.setdiff1d(sample, representatives)
                drawn = np.random.default_rng(seed).choice(pool, 5000 - k, replace=False)
                sample = np.concatenate([representatives, drawn])
            metrics = {
                "k": k,
                "silhouette": float(silhouette_score(projected[sample], labels[sample])),
                "davies_bouldin": float(davies_bouldin_score(projected, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(projected, labels)),
                "smallest_cluster": int(sizes.min()),
                "cluster_sizes": sizes.tolist(),
            }
            results.append((metrics, labels))

    if not results:
        if clusters is not None:
            raise BlockError(f"Embeddings do not support {clusters} distinct visual clusters")
        return [0] * len(features), projected, [], 1
    feasible = [item for item in results if item[0]["smallest_cluster"] >= min_cluster_size]
    if feasible:
        metrics, labels = max(
            feasible, key=lambda item: (item[0]["silhouette"], -item[0]["davies_bouldin"])
        )
    else:
        metrics, labels = max(
            results,
            key=lambda item: (
                item[0]["smallest_cluster"],
                item[0]["silhouette"],
                -item[0]["davies_bouldin"],
            ),
        )
    return labels, projected, results, metrics["k"]


def _umap(features, seed):
    if len(features) < 5:
        raise BlockError("UMAP projection requires at least five images")
    try:
        from umap import UMAP
    except ImportError as error:
        raise BlockError("Install UMAP with uv sync --extra visual --extra umap") from error
    return UMAP(n_neighbors=min(15, len(features) - 1), random_state=seed, n_jobs=1).fit_transform(
        features
    )
