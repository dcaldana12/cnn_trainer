"""Dataset validation block."""

from __future__ import annotations

import json
import math
from collections import Counter

from ..context import Context
from ..dataset import Dataset
from ..errors import BlockError
from ..registry import block


@block(
    "validate",
    "Check labels, class ids, box geometry, and image/label pairing.",
    aliases=("check_dataset",),
)
def validate(
    dataset: Dataset,
    ctx: Context,
    *,
    num_classes: int | None = None,
    strict: bool = True,
    check_bounds: bool = True,
    require_labels: bool = True,
    orphan_labels: str = "warning",
    write_report: bool = True,
    max_logged: int = 20,
) -> Dataset:
    errors: list[str] = []
    warnings: list[str] = []

    if orphan_labels not in {"ignore", "warning", "error"}:
        raise BlockError("validate.orphan_labels must be one of: ignore, warning, error")

    if num_classes is not None:
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 0:
            raise BlockError("validate.num_classes must be a positive integer")
        if num_classes < len(dataset.classes):
            errors.append(
                f"num_classes={num_classes} is smaller than the dataset's "
                f"{len(dataset.classes)} classes"
            )
        elif num_classes > len(dataset.classes):
            start = len(dataset.classes)
            dataset.classes.extend(f"class_{index}" for index in range(start, num_classes))
            warnings.append(
                f"Added {num_classes - start} placeholder class name(s) to reach "
                f"num_classes={num_classes}"
            )

    errors.extend(str(message) for message in dataset.metadata.get("parse_errors", []))

    valid_images = 0
    total_annotations = 0

    for record in dataset.images:
        error_count = len(errors)
        relative = record.relative_path.as_posix()

        if not record.has_label and require_labels:
            errors.append(f"{relative}: missing label file")

        if record.width <= 0 or record.height <= 0:
            errors.append(f"{relative}: invalid image size {record.width}x{record.height}")

        tolerance = 1e-6 * max(record.width, record.height, 1)
        for index, annotation in enumerate(record.annotations, start=1):
            total_annotations += 1
            prefix = f"{relative}, annotation {index}"

            if not 0 <= annotation.class_id < len(dataset.classes):
                errors.append(
                    f"{prefix}: class id {annotation.class_id} is outside "
                    f"0-{max(len(dataset.classes) - 1, 0)}"
                )

            values = (annotation.x1, annotation.y1, annotation.x2, annotation.y2)
            if not all(math.isfinite(value) for value in values):
                errors.append(f"{prefix}: box contains NaN or infinity")
                continue

            if annotation.x1 >= annotation.x2 or annotation.y1 >= annotation.y2:
                errors.append(f"{prefix}: box has zero or negative width/height")
                continue

            if check_bounds and record.width > 0 and record.height > 0:
                if (
                    annotation.x1 < -tolerance
                    or annotation.y1 < -tolerance
                    or annotation.x2 > record.width + tolerance
                    or annotation.y2 > record.height + tolerance
                ):
                    errors.append(
                        f"{prefix}: box extends outside image bounds "
                        f"({record.width}x{record.height})"
                    )

        if len(errors) == error_count:
            valid_images += 1

    collisions = dataset.label_collisions()
    if collisions:
        examples = ", ".join(collisions[:5])
        errors.append(
            f"{len(collisions)} label stem collision(s) would overwrite files on "
            f"YOLO export. Examples: {examples}"
        )

    relative_counts = Counter(record.relative_path.as_posix() for record in dataset.images)
    duplicates = sorted(path for path, count in relative_counts.items() if count > 1)
    if duplicates:
        errors.append(
            f"{len(duplicates)} duplicate relative image path(s): {', '.join(duplicates[:5])}"
        )

    if dataset.orphans:
        messages = [f"orphan label: {path}" for path in dataset.orphans]
        if orphan_labels == "error":
            errors.extend(messages)
        elif orphan_labels == "warning":
            warnings.extend(messages)

    report = {
        "ok": not errors,
        "images": len(dataset.images),
        "valid_images": valid_images,
        "invalid_images": len(dataset.images) - valid_images,
        "annotations": total_annotations,
        "errors": errors,
        "warnings": warnings,
        "orphan_labels": [str(path) for path in dataset.orphans],
        "label_collisions": collisions,
    }

    if write_report:
        ctx.ensure_workdir()
        report_path = ctx.workdir / "validation.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        ctx.logger.info("Wrote %s", report_path)

    ctx.logger.info(
        "Validation: %d images (%d valid), %d annotations, %d error(s), %d warning(s)",
        report["images"],
        report["valid_images"],
        report["annotations"],
        len(errors),
        len(warnings),
    )
    for message in errors[:max_logged]:
        ctx.logger.error("  %s", message)
    if len(errors) > max_logged:
        ctx.logger.error("  ... and %d more error(s)", len(errors) - max_logged)
    for message in warnings[:max_logged]:
        ctx.logger.warning("  %s", message)
    if len(warnings) > max_logged:
        ctx.logger.warning("  ... and %d more warning(s)", len(warnings) - max_logged)

    if errors and strict:
        raise BlockError(f"Dataset validation failed with {len(errors)} error(s)")

    return dataset
