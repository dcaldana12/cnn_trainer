"""Train/validation/test split block."""

from __future__ import annotations

import math
import random
from collections import Counter

from ..context import Context
from ..dataset import KNOWN_SPLITS, Dataset, ImageRecord
from ..errors import BlockError
from ..registry import block

_SPLIT_PRIORITY = {"train": 0, "val": 1, "test": 2}
_DEFAULT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}


@block("split", "Assign images to train, validation and test splits.")
def split(
    dataset: Dataset,
    ctx: Context,
    *,
    ratios: dict | None = None,
    seed: int | None = None,
    stratify: bool | str = True,
    shuffle: bool = True,
    overwrite: bool = True,
) -> Dataset:
    ratios = _normalize_ratios(ratios)
    seed = ctx.seed if seed is None else int(seed)
    stratify = _normalize_bool(stratify, "stratify")

    if not overwrite and any(record.split is not None for record in dataset.images):
        raise BlockError("Dataset already has split assignments. Set overwrite: true to re-split.")

    records = sorted(dataset.images, key=lambda record: record.relative_path.as_posix())
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(records)

    for record in records:
        record.split = None

    if stratify:
        groups: dict[str, list[ImageRecord]] = {}
        for record in records:
            groups.setdefault(_dominant_class(record), []).append(record)
        for key in sorted(groups):
            _assign(groups[key], ratios)
    else:
        _assign(records, ratios)

    counts = Counter(record.split for record in dataset.images)
    ctx.logger.info(
        "Split %d images: %s (seed=%d, stratified=%s)",
        len(dataset.images),
        {name: counts.get(name, 0) for name in KNOWN_SPLITS if ratios.get(name, 0) > 0},
        seed,
        stratify,
    )
    return dataset


def _normalize_ratios(ratios: dict | None) -> dict[str, float]:
    if ratios is None:
        ratios = dict(_DEFAULT_RATIOS)
    if not isinstance(ratios, dict):
        raise BlockError("split.ratios must be a mapping of split name to fraction")

    unknown = sorted(set(ratios) - set(KNOWN_SPLITS))
    if unknown:
        raise BlockError(
            f"Unknown split name(s): {', '.join(unknown)}. Allowed: {', '.join(KNOWN_SPLITS)}"
        )

    values: dict[str, float] = {}
    for name in KNOWN_SPLITS:
        raw = ratios.get(name, 0)
        try:
            value = float(raw)
        except (TypeError, ValueError) as error:
            raise BlockError(f"Invalid ratio for {name!r}: {raw!r}") from error
        if value < 0:
            raise BlockError(f"Ratio for {name!r} must be non-negative")
        values[name] = value

    total = sum(values.values())
    if total <= 0:
        raise BlockError("At least one split ratio must be greater than zero")

    return {name: value / total for name, value in values.items()}


def _normalize_bool(value, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "class", "classes"}:
            return True
        if lowered in {"false", "no", "0", "none"}:
            return False
    raise BlockError(f"{name} must be true or false, got {value!r}")


def _assign(records: list[ImageRecord], ratios: dict[str, float]) -> None:
    counts = _allocate(len(records), ratios)
    start = 0
    for name in KNOWN_SPLITS:
        end = start + counts[name]
        for record in records[start:end]:
            record.split = name
        start = end


def _allocate(total: int, ratios: dict[str, float]) -> dict[str, int]:
    quotas = {name: total * ratio for name, ratio in ratios.items()}
    counts = {name: math.floor(quota) for name, quota in quotas.items()}
    remainder = total - sum(counts.values())

    order = sorted(
        ratios,
        key=lambda name: (quotas[name] - counts[name], -_SPLIT_PRIORITY[name]),
        reverse=True,
    )
    for name in order:
        if remainder <= 0:
            break
        counts[name] += 1
        remainder -= 1

    while remainder > 0:
        for name in order:
            if remainder <= 0:
                break
            counts[name] += 1
            remainder -= 1

    return counts


def _dominant_class(record: ImageRecord) -> str:
    if not record.annotations:
        return "background"
    counts = Counter(annotation.class_id for annotation in record.annotations)
    class_id = max(counts, key=lambda value: (counts[value], -value))
    return f"class_{class_id}"
