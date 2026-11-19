"""Blocks that change labels and class definitions."""

from __future__ import annotations

from collections import Counter

from ..context import Context
from ..dataset import Dataset
from ..errors import BlockError
from ..registry import block


@block(
    "create_missing_labels",
    "Mark images without label files as background images.",
    aliases=("create_empty_label_file",),
)
def create_missing_labels(dataset: Dataset, ctx: Context) -> Dataset:
    """Mark missing labels as valid empty (background) labels.

    Files are not created on disk right away; the YOLO export block writes an
    empty ``.txt`` for every background image.
    """
    created = 0
    for record in dataset.images:
        if not record.has_label:
            record.has_label = True
            created += 1

    ctx.logger.info(
        "Marked %d image(s) without label files as background; "
        "empty labels will be written on export.",
        created,
    )
    return dataset


@block(
    "remap_classes",
    "Merge, rename, or drop classes using old -> new mappings.",
    aliases=("change_label", "change_labels"),
)
def remap_classes(
    dataset: Dataset,
    ctx: Context,
    *,
    mapping: dict,
    prune_unused: bool = False,
) -> Dataset:
    """Apply a class mapping.

    Keys and values may be class indices or class names. A value of ``null``
    drops the annotations. A value that is a new name renames the source class
    in place; a value that matches an existing class merges into it.
    """
    if not isinstance(mapping, dict):
        raise BlockError("remap_classes expects 'mapping' to be a mapping")

    targets = _resolve_mapping(dataset.classes, mapping)
    moved = 0
    dropped = 0

    for record in dataset.images:
        kept = []
        for annotation in record.annotations:
            if annotation.class_id in targets:
                target = targets[annotation.class_id]
                if target is None:
                    dropped += 1
                    continue
                if target != annotation.class_id:
                    annotation.class_id = target
                    moved += 1
            kept.append(annotation)
        record.annotations = kept

    if prune_unused:
        _prune_unused_classes(dataset, ctx)

    ctx.logger.info(
        "Applied class mapping: %d annotation(s) moved, %d dropped; classes: %s",
        moved,
        dropped,
        dataset.classes,
    )
    return dataset


@block(
    "remove_class",
    "Remove every annotation belonging to one or more classes.",
    aliases=("remove_label",),
)
def remove_class(
    dataset: Dataset,
    ctx: Context,
    *,
    classes,
    prune_unused: bool = False,
) -> Dataset:
    targets = classes if isinstance(classes, (list, tuple, set)) else [classes]
    targets = list(targets)
    if not targets:
        raise BlockError("remove_class needs at least one class id or name")

    mapping = {target: None for target in targets}
    return remap_classes(dataset, ctx, mapping=mapping, prune_unused=prune_unused)


def _resolve_mapping(classes: list[str], mapping: dict) -> dict[int, int | None]:
    resolved_keys: list[tuple[int, object]] = []
    seen: set[int] = set()
    for key, value in mapping.items():
        source = _resolve_source_ref(key, classes)
        if source in seen:
            raise BlockError(f"Class {source} ({classes[source]!r}) is mapped more than once")
        seen.add(source)
        resolved_keys.append((source, value))

    targets: dict[int, int | None] = {}
    for source, value in resolved_keys:
        if value is None:
            target = None
        elif isinstance(value, bool):
            raise BlockError(f"Invalid target for class {key!r}: {value!r}")
        elif isinstance(value, int):
            if not 0 <= value < len(classes):
                raise BlockError(
                    f"Target class {value} for {key!r} is outside 0-{len(classes) - 1}"
                )
            target = value
        elif isinstance(value, str):
            name = value.strip()
            if not name:
                raise BlockError(f"Empty target class name for {key!r}")
            if name in classes:
                target = classes.index(name)
            else:
                try:
                    index = int(name)
                except ValueError:
                    index = None
                if index is not None:
                    if not 0 <= index < len(classes):
                        raise BlockError(
                            f"Target class {index} for {key!r} is outside 0-{len(classes) - 1}"
                        )
                    target = index
                else:
                    # Rename the source class in place. If another mapping
                    # already introduced this name, the lookup above merges it.
                    classes[source] = name
                    target = source
        else:
            raise BlockError(f"Invalid target for class {key!r}: {value!r}")

        targets[source] = target
    return targets


def _resolve_source_ref(value, classes: list[str]) -> int:
    if isinstance(value, bool):
        raise BlockError(f"Invalid class reference: {value!r}")
    if isinstance(value, int):
        if 0 <= value < len(classes):
            return value
        raise BlockError(f"Class id {value} is outside 0-{len(classes) - 1}")
    if isinstance(value, str):
        if value in classes:
            return classes.index(value)
        try:
            index = int(value)
        except ValueError as error:
            raise BlockError(f"Unknown class: {value!r}") from error
        if 0 <= index < len(classes):
            return index
        raise BlockError(f"Class id {index} is outside 0-{len(classes) - 1}")
    raise BlockError(f"Invalid class reference: {value!r}")


def _prune_unused_classes(dataset: Dataset, ctx: Context) -> None:
    if not dataset.classes:
        return

    counts = Counter(
        annotation.class_id for record in dataset.images for annotation in record.annotations
    )
    keep = [index for index in range(len(dataset.classes)) if counts[index] > 0]
    if len(keep) == len(dataset.classes):
        return

    removed = len(dataset.classes) - len(keep)
    old_to_new = {old: new for new, old in enumerate(keep)}
    dataset.classes = [dataset.classes[index] for index in keep]
    for record in dataset.images:
        for annotation in record.annotations:
            annotation.class_id = old_to_new[annotation.class_id]

    ctx.logger.info(
        "Pruned %d unused class(es); remaining classes: %s",
        removed,
        dataset.classes,
    )
