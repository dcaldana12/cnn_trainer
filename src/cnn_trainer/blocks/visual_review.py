"""Write plots and representative image sheets for visual split review."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import mkdtemp

from ..errors import BlockError

_METRICS = (
    ("silhouette", "Silhouette"),
    ("davies_bouldin", "Davies–Bouldin"),
    ("calinski_harabasz", "Calinski–Harabasz"),
    ("smallest_cluster", "Smallest cluster (images)"),
)


def write_visual_review(
    ctx, records, candidates, projected, coordinates, chosen, backbone, minimum, sample_count
) -> Path:
    try:
        import numpy as np
        from matplotlib.figure import Figure
        from PIL import Image, ImageOps
    except ImportError as error:
        raise BlockError("Install review dependencies with uv sync --extra visual") from error

    directory = Path(mkdtemp(prefix="visual-review-", dir=ctx.ensure_workdir()))
    metrics = [entry[0] for entry in candidates]
    report = {
        "backbone": backbone,
        "recommended_k": chosen if metrics else None,
        "min_cluster_size": minimum,
        "candidates": metrics,
    }
    if not metrics:
        ctx.logger.warning("Embeddings did not support two or more distinct clusters")
    write_k_scores(
        directory / "k-scores.png", metrics, chosen, backbone, minimum, label="Recommended"
    )

    for item, labels in candidates:
        k = item["k"]
        folder = directory / f"k-{k:02d}"
        folder.mkdir()
        _plot_candidate(Figure, folder, k, labels, item["cluster_sizes"], projected, coordinates)
        # Choose examples closest to each cluster mean in the PCA space used by K-Means.
        members = []
        for cluster in range(k):
            indices = np.flatnonzero(labels == cluster)
            vectors = projected[indices]
            distances = np.sum((vectors - vectors.mean(axis=0)) ** 2, axis=1)
            order = np.argsort(distances, kind="stable")[:sample_count]
            members.append(indices[order].tolist())
        (folder / "samples.json").write_text(
            json.dumps(
                [[records[index].relative_path.as_posix() for index in row] for row in members],
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        _plot_samples(Figure, Image, ImageOps, folder, records, members, sample_count)
    (directory / "review.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return directory


def write_k_scores(
    path: Path,
    metrics: list[dict],
    chosen: int,
    backbone: str,
    minimum: int,
    *,
    label: str = "Selected",
) -> None:
    try:
        from matplotlib.figure import Figure
    except ImportError as error:
        raise BlockError("Install plot dependencies with uv sync --extra visual") from error

    scores = Figure(figsize=(10, 7), layout="constrained")
    if metrics:
        candidates = [item["k"] for item in metrics]
        for ax, (key, title) in zip(scores.subplots(2, 2).flat, _METRICS, strict=True):
            ax.plot(candidates, [item[key] for item in metrics], "o-")
            ax.axvline(chosen, color="0.45", linestyle="--")
            if key == "smallest_cluster":
                ax.axhline(minimum, color="tab:red", linestyle=":")
            if len(candidates) <= 12:
                ax.set_xticks(candidates)
            ax.set(xlabel="K", title=title)
        if any(item["smallest_cluster"] >= minimum for item in metrics):
            reason = "Ranked by silhouette, then Davies–Bouldin, among eligible K values"
        else:
            reason = "No K met the minimum cluster size. Ranked by largest smallest cluster first"
    else:
        reason = "No candidate produced two or more distinct clusters. Using one cluster"
        scores.text(0.5, 0.5, reason, ha="center")
    title = (
        f"{backbone} · {label} K = {chosen}" if metrics else f"{backbone} · One-cluster fallback"
    )
    scores.suptitle(f"{title}\n{reason}" if metrics else title)
    scores.savefig(path, dpi=140)


def _plot_candidate(Figure, folder, k, labels, sizes, projected, coordinates):
    panels = [("PCA", projected)]
    if coordinates is not None:
        panels.append(("UMAP", coordinates))
    figure = Figure(figsize=(5 * (len(panels) + 1), 4), layout="constrained")
    axes = figure.subplots(1, len(panels) + 1)
    for ax, (name, points) in zip(axes[:-1], panels, strict=True):
        dots = ax.scatter(
            points[:, 0],
            points[:, 1] if points.shape[1] > 1 else 0 * points[:, 0],
            c=labels,
            cmap="tab20",
            vmin=0,
            vmax=max(1, k - 1),
            s=9,
        )
        bar = figure.colorbar(dots, ax=ax)
        if k <= 20:
            bar.set_ticks(range(k))
        ax.set(xlabel=f"{name} 1", ylabel=f"{name} 2", title=f"{name} projection")
    axes[-1].bar(range(k), sizes)
    axes[-1].set(xlabel="Cluster", ylabel="Images", title="Cluster sizes")
    figure.suptitle(f"K = {k}")
    figure.savefig(folder / "clusters.png", dpi=140)


def _plot_samples(Figure, Image, ImageOps, folder, records, members, sample_count):
    images = folder / "images"
    images.mkdir()
    # Paginate contact sheets so a large K cannot produce arbitrarily tall images.
    for start in range(0, len(members), 8):
        page = members[start : start + 8]
        columns = min(sample_count, max(map(len, page)))
        figure = Figure(figsize=(2.75 * columns, 2.1 * len(page)), layout="constrained")
        axes = figure.subplots(len(page), columns, squeeze=False)
        for row, indices in enumerate(page):
            for column, ax in enumerate(axes[row]):
                ax.axis("off")
                if column >= len(indices):
                    continue
                record = records[indices[column]]
                with Image.open(record.path) as source:
                    source.draft("RGB", (320, 240))
                    image = ImageOps.exif_transpose(source)
                    image.thumbnail((320, 240))
                    preview = image.convert("RGB")
                preview.save(images / f"cluster-{start + row:02d}-{column + 1:02d}.jpg")
                ax.imshow(preview)
                if column == 0:
                    ax.set_title(f"Cluster {start + row}", loc="left", fontsize=9)
        figure.savefig(folder / f"samples-{start // 8 + 1:02d}.png", dpi=120)
