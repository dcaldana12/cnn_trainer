"""Dataset quality audit and actionable suggestions."""

from __future__ import annotations

import json
import statistics
from collections import Counter

from ..context import Context
from ..dataset import KNOWN_SPLITS, Dataset
from ..errors import BlockError
from ..registry import block

_LEVEL_ORDER = {"warning": 0, "tip": 1}


@block(
    "audit_dataset",
    "Report class balance, split coverage, box quality and dataset tips.",
    aliases=("analyze", "dataset_audit", "insights"),
)
def audit_dataset(
    dataset: Dataset,
    ctx: Context,
    *,
    min_instances_per_class: int = 20,
    min_images_per_class: int = 10,
    max_imbalance_ratio: float = 10.0,
    tiny_box_area: float = 0.0005,
    max_tiny_share: float = 0.1,
    border_margin: float = 0.01,
    min_image_side: int = 64,
    max_aspect_ratio: float = 4.0,
    min_val_images: int = 20,
    write_report: bool = True,
    write_markdown: bool = True,
    max_logged: int = 12,
) -> Dataset:
    """Analyze *dataset* in memory and write ``audit.json`` / ``audit.md``.

    The block never modifies images or labels. It records the report in
    ``dataset.metadata["audit"]`` and prints the most important tips, so it can
    be placed anywhere in a pipeline.
    """
    _validate_options(
        min_instances_per_class=min_instances_per_class,
        min_images_per_class=min_images_per_class,
        max_imbalance_ratio=max_imbalance_ratio,
        tiny_box_area=tiny_box_area,
        max_tiny_share=max_tiny_share,
        border_margin=border_margin,
        min_image_side=min_image_side,
        max_aspect_ratio=max_aspect_ratio,
        min_val_images=min_val_images,
        max_logged=max_logged,
    )
    if not dataset.images:
        raise BlockError("Cannot audit an empty dataset")

    class_instances = {name: 0 for name in dataset.classes}
    images_per_class = {name: 0 for name in dataset.classes}
    class_by_split = {name: {split: 0 for split in KNOWN_SPLITS} for name in dataset.classes}
    split_counts: Counter[str] = Counter()
    objects_per_image: list[int] = []
    box_areas: list[float] = []
    widths: list[int] = []
    heights: list[int] = []
    empty_images = 0
    missing_labels = 0
    border_boxes = 0
    tiny_boxes = 0
    invalid_boxes = 0
    annotation_total = 0

    for record in dataset.images:
        split_counts[record.split or "unassigned"] += 1
        widths.append(record.width)
        heights.append(record.height)
        objects_per_image.append(len(record.annotations))
        if not record.annotations:
            empty_images += 1
        if not record.has_label:
            missing_labels += 1

        present_classes: set[int] = set()
        for annotation in record.annotations:
            annotation_total += 1
            present_classes.add(annotation.class_id)
            if not 0 <= annotation.class_id < len(dataset.classes):
                invalid_boxes += 1
                continue
            name = dataset.classes[annotation.class_id]
            class_instances[name] += 1
            if record.split in class_by_split[name]:
                class_by_split[name][record.split] += 1

            box_width = max(0.0, annotation.x2 - annotation.x1)
            box_height = max(0.0, annotation.y2 - annotation.y1)
            image_area = max(1, record.width * record.height)
            area_ratio = box_width * box_height / image_area
            box_areas.append(area_ratio)

            if area_ratio < tiny_box_area:
                tiny_boxes += 1
            margin_x = record.width * border_margin
            margin_y = record.height * border_margin
            if (
                annotation.x1 <= margin_x
                or annotation.y1 <= margin_y
                or annotation.x2 >= record.width - margin_x
                or annotation.y2 >= record.height - margin_y
            ):
                border_boxes += 1

        for class_id in present_classes:
            if 0 <= class_id < len(dataset.classes):
                images_per_class[dataset.classes[class_id]] += 1

    has_splits = any(record.split is not None for record in dataset.images)
    issues = _build_issues(
        dataset=dataset,
        class_instances=class_instances,
        images_per_class=images_per_class,
        class_by_split=class_by_split,
        split_counts=split_counts,
        empty_images=empty_images,
        missing_labels=missing_labels,
        tiny_boxes=tiny_boxes,
        border_boxes=border_boxes,
        invalid_boxes=invalid_boxes,
        annotation_total=annotation_total,
        box_areas=box_areas,
        widths=widths,
        heights=heights,
        has_splits=has_splits,
        min_instances_per_class=min_instances_per_class,
        min_images_per_class=min_images_per_class,
        max_imbalance_ratio=max_imbalance_ratio,
        tiny_box_area=tiny_box_area,
        max_tiny_share=max_tiny_share,
        min_image_side=min_image_side,
        max_aspect_ratio=max_aspect_ratio,
        min_val_images=min_val_images,
    )

    report = {
        "images": len(dataset.images),
        "annotations": annotation_total,
        "classes": len(dataset.classes),
        "class_instances": class_instances,
        "images_per_class": images_per_class,
        "class_by_split": class_by_split if has_splits else {},
        "splits": {
            split: split_counts.get(split, 0)
            for split in KNOWN_SPLITS
            if split_counts.get(split, 0)
        },
        "unassigned": split_counts.get("unassigned", 0),
        "empty_images": empty_images,
        "missing_labels": missing_labels,
        "invalid_annotations": invalid_boxes,
        "objects_per_image": {
            "min": min(objects_per_image),
            "median": float(statistics.median(objects_per_image)),
            "mean": round(statistics.fmean(objects_per_image), 2),
            "max": max(objects_per_image),
            "p95": _percentile(objects_per_image, 0.95),
        },
        "box_area_ratio": {
            "min": round(min(box_areas), 6) if box_areas else None,
            "median": round(float(statistics.median(box_areas)), 6) if box_areas else None,
            "mean": round(statistics.fmean(box_areas), 6) if box_areas else None,
            "max": round(max(box_areas), 6) if box_areas else None,
            "tiny": tiny_boxes,
            "tiny_share": round(tiny_boxes / annotation_total, 4) if annotation_total else 0.0,
            "near_border": border_boxes,
        },
        "image_sizes": {
            "min_width": min(widths),
            "min_height": min(heights),
            "max_width": max(widths),
            "max_height": max(heights),
            "small": sum(
                1
                for width, height in zip(widths, heights, strict=False)
                if min(width, height) < min_image_side
            ),
            "extreme_aspect": sum(
                1
                for width, height in zip(widths, heights, strict=False)
                if max(width, height) / max(1, min(width, height)) > max_aspect_ratio
            ),
        },
        "issues": issues,
        "tips": [issue["message"] for issue in issues],
    }

    redundancy = dataset.metadata.get("redundancy")
    if isinstance(redundancy, dict):
        candidates = redundancy.get("candidates") or []
        report["redundancy"] = {
            "candidates": len(candidates),
            "counts": redundancy.get("counts", {}),
        }

    dataset.metadata["audit"] = report

    if write_report:
        destination = ctx.ensure_workdir() / "audit.json"
        destination.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")
    if write_markdown:
        markdown = _render_markdown(report)
        (ctx.ensure_workdir() / "audit.md").write_text(markdown, encoding="utf-8")

    warnings = sum(1 for issue in issues if issue["level"] == "warning")
    ctx.logger.info(
        "Audit: %d images, %d annotations, %d class(es); %d warning(s), %d tip(s)",
        report["images"],
        report["annotations"],
        report["classes"],
        warnings,
        len(issues) - warnings,
    )
    for issue in sorted(issues, key=lambda item: _LEVEL_ORDER[item["level"]])[:max_logged]:
        ctx.logger.info("  %s: %s", issue["level"], issue["message"])
    if write_report:
        ctx.logger.info("Wrote %s", ctx.workdir / "audit.json")
    return dataset


def _build_issues(
    *,
    dataset: Dataset,
    class_instances: dict[str, int],
    images_per_class: dict[str, int],
    class_by_split: dict[str, dict[str, int]],
    split_counts: Counter,
    empty_images: int,
    missing_labels: int,
    tiny_boxes: int,
    border_boxes: int,
    invalid_boxes: int,
    annotation_total: int,
    box_areas: list[float],
    widths: list[int],
    heights: list[int],
    has_splits: bool,
    min_instances_per_class: int,
    min_images_per_class: int,
    max_imbalance_ratio: float,
    tiny_box_area: float,
    max_tiny_share: float,
    min_image_side: int,
    max_aspect_ratio: float,
    min_val_images: int,
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []

    def add(level: str, message: str) -> None:
        issues.append({"level": level, "message": message})

    if not dataset.classes:
        add("warning", "No classes are defined; check the source metadata or class_names.")

    zero_classes = [name for name, count in class_instances.items() if count == 0]
    if zero_classes:
        add(
            "warning",
            f"Class(es) without annotations: {', '.join(zero_classes)}. "
            "Remove them with remove_class or collect examples.",
        )

    rare = [
        (name, count)
        for name, count in class_instances.items()
        if 0 < count < min_instances_per_class
    ]
    if rare:
        detail = ", ".join(f"{name}={count}" for name, count in rare)
        add(
            "warning",
            f"Rare class(es) with fewer than {min_instances_per_class} instances: "
            f"{detail}. Collect more data or merge/remove them.",
        )

    few_images = [
        (name, count)
        for name, count in images_per_class.items()
        if 0 < count < min_images_per_class
    ]
    if few_images:
        detail = ", ".join(f"{name}={count}" for name, count in few_images)
        add(
            "tip",
            f"Class(es) present in fewer than {min_images_per_class} images: {detail}. "
            "Object detectors need varied contexts, not only many boxes.",
        )

    counts = [count for count in class_instances.values() if count > 0]
    if len(counts) >= 2:
        ratio = max(counts) / min(counts)
        if ratio > max_imbalance_ratio:
            add(
                "warning",
                f"Class imbalance is {ratio:.1f}:1 (largest {max(counts)}, smallest "
                f"{min(counts)}). Consider more data for rare classes, remap_classes, "
                "or class-aware sampling.",
            )

    if missing_labels:
        add(
            "warning",
            f"{missing_labels} image(s) have no label file. Run create_missing_labels "
            "if they are intentional background, otherwise fix the source annotations.",
        )
    if empty_images:
        add(
            "tip",
            f"{empty_images} image(s) contain no boxes. A few background images help "
            "reduce false positives, but review them and remove uninformative ones.",
        )

    if annotation_total:
        tiny_share = tiny_boxes / annotation_total
        if tiny_boxes:
            level = "warning" if tiny_share > max_tiny_share else "tip"
            add(
                level,
                f"{tiny_boxes} box(es) ({tiny_share:.1%}) cover less than "
                f"{tiny_box_area:.2%} of their image. Small objects are hard at "
                "640 px; consider a larger imgsz or excluding them.",
            )
        if invalid_boxes:
            add(
                "warning",
                f"{invalid_boxes} annotation(s) reference a class id outside the class "
                "list; run validate for details.",
            )
    if border_boxes:
        add(
            "tip",
            f"{border_boxes} box(es) touch the image border. Check that they are not "
            "truncated objects; augmenting crops or adding padding can help.",
        )

    if box_areas:
        small_boxes = sum(1 for area in box_areas if area < 0.01)
        if small_boxes and small_boxes / len(box_areas) > 0.5:
            add(
                "tip",
                "More than half of the boxes cover under 1% of their image. "
                "Training at a larger imgsz may improve small-object recall.",
            )

    if min(widths) < min_image_side or min(heights) < min_image_side:
        add(
            "warning",
            f"Some images are smaller than {min_image_side} px on a side. They are "
            "upscaled before training and may add little detail; consider removing them.",
        )
    extreme = sum(
        1
        for width, height in zip(widths, heights, strict=False)
        if max(width, height) / max(1, min(width, height)) > max_aspect_ratio
    )
    if extreme:
        add(
            "tip",
            f"{extreme} image(s) have an aspect ratio above {max_aspect_ratio}:1. "
            "Letterboxing or padding avoids heavy distortion.",
        )

    if has_splits:
        unassigned = split_counts.get("unassigned", 0)
        if unassigned:
            add(
                "warning",
                f"{unassigned} image(s) have no split assignment; they fall back to "
                "the default split during export.",
            )
        val_count = split_counts.get("val", 0)
        if val_count == 0:
            add("warning", "No validation images. Ultralytics training requires a val split.")
        elif val_count < min_val_images:
            add(
                "tip",
                f"Only {val_count} validation image(s). Metrics will be noisy; aim for "
                f"at least {min_val_images}.",
            )
        if split_counts.get("test", 0) == 0:
            add("tip", "No test split. A held-out test set gives a less biased final score.")
        for name, per_split in class_by_split.items():
            if class_instances.get(name, 0) == 0:
                continue
            missing_val = per_split.get("val", 0) == 0
            missing_test = per_split.get("test", 0) == 0
            if missing_val:
                add(
                    "warning",
                    f"Class {name!r} has no instances in val; validation cannot measure it.",
                )
            elif missing_test:
                add(
                    "tip",
                    f"Class {name!r} has no instances in test; the final score misses it.",
                )
    else:
        add("tip", "No split assignments yet; add a split block before export or training.")

    redundancy = dataset.metadata.get("redundancy")
    if isinstance(redundancy, dict):
        candidates = redundancy.get("candidates") or []
        if candidates:
            add(
                "tip",
                f"{len(candidates)} near-duplicate or blank image candidate(s) were "
                "found. Review them and run remove_flagged_images if they add no value.",
            )
        cross_split = redundancy.get("cross_split_groups", 0)
        if cross_split and has_splits:
            add(
                "warning",
                f"{cross_split} near-duplicate group(s) span different splits. "
                "Validation metrics can be inflated; remove duplicates or rerun "
                "find_redundant_images before split.",
            )

    return issues


def _validate_options(
    *,
    min_instances_per_class: int,
    min_images_per_class: int,
    max_imbalance_ratio: float,
    tiny_box_area: float,
    max_tiny_share: float,
    border_margin: float,
    min_image_side: int,
    max_aspect_ratio: float,
    min_val_images: int,
    max_logged: int,
) -> None:
    for name, value in (
        ("min_instances_per_class", min_instances_per_class),
        ("min_images_per_class", min_images_per_class),
        ("min_image_side", min_image_side),
        ("min_val_images", min_val_images),
        ("max_logged", max_logged),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BlockError(f"{name} must be a non-negative integer")
    if not 0 < tiny_box_area <= 1:
        raise BlockError("tiny_box_area must be between 0 and 1")
    if not 0 <= max_tiny_share <= 1:
        raise BlockError("max_tiny_share must be between 0 and 1")
    if not 0 <= border_margin < 0.5:
        raise BlockError("border_margin must be between 0 and 0.5")
    if max_imbalance_ratio <= 0:
        raise BlockError("max_imbalance_ratio must be greater than zero")
    if max_aspect_ratio < 1:
        raise BlockError("max_aspect_ratio must be at least 1")


def _percentile(values: list[int], fraction: float) -> int | float:
    ordered = sorted(values)
    index = round(fraction * (len(ordered) - 1))
    return ordered[max(0, min(len(ordered) - 1, index))]


def _render_markdown(report: dict) -> str:
    splits = report["splits"]
    split_text = ", ".join(f"{name}={count}" for name, count in splits.items())
    if report["unassigned"]:
        split_text = (
            f"{split_text}, unassigned={report['unassigned']}"
            if split_text
            else f"unassigned={report['unassigned']}"
        )
    objects = report["objects_per_image"]
    areas = report["box_area_ratio"]
    lines = [
        "# Dataset audit",
        "",
        f"- Images: {report['images']}",
        f"- Annotations: {report['annotations']}",
        f"- Classes: {report['classes']}",
        f"- Splits: {split_text or 'none'}",
        f"- Empty images: {report['empty_images']}",
        f"- Missing label files: {report['missing_labels']}",
        f"- Objects per image: min {objects['min']}, median {objects['median']}, "
        f"mean {objects['mean']}, max {objects['max']}",
        f"- Tiny boxes: {areas['tiny']} ({areas['tiny_share']:.1%}); "
        f"border boxes: {areas['near_border']}",
        "",
        "## Classes",
        "",
        "| Class | Instances | Images | Train | Val | Test |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    has_split_data = bool(report["class_by_split"])
    for name in report["class_instances"]:
        per_split = report["class_by_split"].get(name, {})
        split_columns = (
            f"{per_split.get('train', 0)} | {per_split.get('val', 0)} | {per_split.get('test', 0)}"
            if has_split_data
            else "- | - | -"
        )
        lines.append(
            f"| {name} | {report['class_instances'][name]} | "
            f"{report['images_per_class'][name]} | {split_columns} |"
        )

    lines.extend(["", "## Findings", ""])
    if report["issues"]:
        for issue in report["issues"]:
            lines.append(f"- **{issue['level']}**: {issue['message']}")
    else:
        lines.append("- No issues found.")

    return "\n".join(lines) + "\n"
