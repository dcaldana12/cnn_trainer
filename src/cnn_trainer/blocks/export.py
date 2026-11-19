"""YOLO export block and reusable writer."""

from __future__ import annotations

import math
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml

from ..context import Context
from ..dataset import KNOWN_SPLITS, Dataset, ImageRecord
from ..errors import BlockError
from ..registry import block
from ..utils import strip_split_prefix

_KNOWN_SPLIT_SET = frozenset(KNOWN_SPLITS)


@block("export_yolo", "Write the dataset in YOLO/Ultralytics directory layout.")
def export_yolo(
    dataset: Dataset,
    ctx: Context,
    *,
    path: str | None = None,
    default_split: str = "train",
    mode: str = "copy",
    overwrite: bool = False,
    clamp: bool = False,
    absolute_path: bool = False,
) -> Dataset:
    destination = ctx.resolve(path) if path else ctx.workdir / "dataset"
    if destination is None:
        raise BlockError("Invalid export path")

    data_path = write_yolo_dataset(
        dataset,
        destination,
        default_split=default_split,
        mode=mode,
        overwrite=overwrite,
        clamp=clamp,
        absolute_path=absolute_path,
    )
    counts = Counter(record.split or default_split for record in dataset.images)
    ctx.logger.info(
        "Exported %d image(s) to %s; splits: %s; data file: %s",
        len(dataset.images),
        destination,
        {name: counts.get(name, 0) for name in KNOWN_SPLITS if counts.get(name)},
        data_path,
    )
    dataset.metadata["last_export"] = str(data_path)
    return dataset


def write_yolo_dataset(
    dataset: Dataset,
    destination: Path,
    *,
    default_split: str = "train",
    mode: str = "copy",
    overwrite: bool = False,
    clamp: bool = False,
    absolute_path: bool = False,
) -> Path:
    """Write *dataset* under *destination* and return the ``data.yaml`` path.

    By default ``data.yaml`` omits ``path`` so Ultralytics uses the YAML file's
    parent directory. This keeps exported datasets and batch ZIPs portable.
    Set ``absolute_path=True`` to pin the dataset root instead.
    """
    if default_split not in KNOWN_SPLITS:
        raise BlockError(
            f"default_split must be one of {', '.join(KNOWN_SPLITS)}, got {default_split!r}"
        )
    if mode not in {"copy", "symlink", "hardlink"}:
        raise BlockError("mode must be one of: copy, symlink, hardlink")

    destination = destination.resolve()
    _prepare_destination(destination, overwrite=overwrite)

    entries: list[tuple[ImageRecord, str, Path]] = []
    seen: dict[tuple[str, str], Path] = {}
    counts: Counter[str] = Counter()

    for record in dataset.images:
        split = record.split or default_split
        if split not in KNOWN_SPLITS:
            raise BlockError(
                f"{record.relative_path}: unknown split {split!r}; "
                f"allowed: {', '.join(KNOWN_SPLITS)}"
            )
        relative = strip_split_prefix(record.relative_path, _KNOWN_SPLIT_SET)
        if not relative.parts:
            raise BlockError(f"{record.relative_path}: empty relative path")

        key = (split, relative.with_suffix("").as_posix())
        if key in seen:
            raise BlockError(
                "Two images would write the same YOLO label file: "
                f"{seen[key]} and {record.path} -> {relative.with_suffix('.txt')}"
            )
        seen[key] = record.path
        entries.append((record, split, relative))
        counts[split] += 1

    if len(set(dataset.classes)) != len(dataset.classes):
        raise BlockError("Class names must be unique before YOLO export")

    for record, split, relative in entries:
        image_destination = destination / "images" / split / relative
        label_destination = destination / "labels" / split / relative.with_suffix(".txt")
        _place_file(record.path, image_destination, mode)
        lines = _label_lines(record, dataset, clamp=clamp)
        label_destination.parent.mkdir(parents=True, exist_ok=True)
        label_destination.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )

    data: dict = {
        "nc": len(dataset.classes),
        "names": {index: name for index, name in enumerate(dataset.classes)},
    }
    if absolute_path:
        data = {"path": str(destination), **data}
    for name in KNOWN_SPLITS:
        if counts.get(name):
            data[name] = f"images/{name}"

    data_path = destination / "data.yaml"
    data_path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return data_path


def _label_lines(record: ImageRecord, dataset: Dataset, *, clamp: bool) -> list[str]:
    if record.width <= 0 or record.height <= 0:
        raise BlockError(
            f"{record.relative_path}: invalid image size {record.width}x{record.height}"
        )

    lines: list[str] = []
    for index, annotation in enumerate(record.annotations, start=1):
        prefix = f"{record.relative_path}, annotation {index}"
        if not 0 <= annotation.class_id < len(dataset.classes):
            raise BlockError(
                f"{prefix}: class id {annotation.class_id} is outside "
                f"0-{max(len(dataset.classes) - 1, 0)}"
            )

        x1, y1, x2, y2 = annotation.x1, annotation.y1, annotation.x2, annotation.y2
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            raise BlockError(f"{prefix}: box contains NaN or infinity")

        if clamp:
            x1 = min(max(x1, 0.0), record.width)
            y1 = min(max(y1, 0.0), record.height)
            x2 = min(max(x2, 0.0), record.width)
            y2 = min(max(y2, 0.0), record.height)

        if x1 >= x2 or y1 >= y2:
            raise BlockError(f"{prefix}: box has zero or negative width/height")

        if not clamp:
            tolerance = 1e-6
            if (
                x1 < -tolerance
                or y1 < -tolerance
                or x2 > record.width + tolerance
                or y2 > record.height + tolerance
            ):
                raise BlockError(
                    f"{prefix}: box extends outside the image; "
                    "run the validate block or set clamp: true"
                )

        x_center = (x1 + x2) / 2.0 / record.width
        y_center = (y1 + y2) / 2.0 / record.height
        width = (x2 - x1) / record.width
        height = (y2 - y1) / record.height
        lines.append(
            f"{annotation.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )
    return lines


def _prepare_destination(destination: Path, *, overwrite: bool) -> None:
    if destination.exists():
        if not destination.is_dir():
            raise BlockError(f"Export path exists and is not a directory: {destination}")
        if any(destination.iterdir()):
            if not overwrite:
                raise BlockError(
                    f"Export directory is not empty: {destination}. "
                    "Set overwrite: true to replace it."
                )
            _safe_rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)


def _safe_rmtree(path: Path) -> None:
    forbidden = {Path(path.anchor), Path.home().resolve(), Path.cwd().resolve()}
    if path in forbidden or len(path.parts) < 3:
        raise BlockError(f"Refusing to remove protected directory: {path}")
    if Path.cwd().resolve().is_relative_to(path):
        raise BlockError(f"Refusing to remove a parent of the current directory: {path}")
    shutil.rmtree(path)


def _place_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()

    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        try:
            os.link(source, destination)
        except OSError as error:
            raise BlockError(
                f"Could not hardlink {source} -> {destination}: {error}. Use mode: copy instead."
            ) from error
    else:
        destination.symlink_to(source.resolve())
