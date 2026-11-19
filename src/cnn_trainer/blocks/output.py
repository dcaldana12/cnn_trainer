"""Output blocks: image lists and batch archives."""

from __future__ import annotations

import shutil

from ..context import Context
from ..dataset import KNOWN_SPLITS, Dataset
from ..errors import BlockError
from ..registry import block
from ..utils import strip_split_prefix
from .export import _prepare_destination, write_yolo_dataset

_KNOWN_SPLIT_SET = frozenset(KNOWN_SPLITS)


@block(
    "write_image_list",
    "Write a text file with one image path per line.",
    aliases=("write_path", "write_train_txt"),
)
def write_image_list(
    dataset: Dataset,
    ctx: Context,
    *,
    path: str,
    split: str | None = None,
    prefix: str | None = None,
    absolute: bool = False,
    default_split: str = "train",
) -> Dataset:
    if split is not None and split not in KNOWN_SPLITS:
        raise BlockError(f"split must be one of {', '.join(KNOWN_SPLITS)}, got {split!r}")

    records = dataset.images
    if split is not None:
        records = [record for record in records if (record.split or default_split) == split]
    if not records:
        raise BlockError("No images matched the requested split")

    lines = []
    for record in records:
        if absolute:
            lines.append(str(record.path))
            continue
        relative = strip_split_prefix(record.relative_path, _KNOWN_SPLIT_SET)
        if prefix:
            lines.append(f"{prefix.rstrip('/')}/{relative.as_posix()}")
        else:
            lines.append(relative.as_posix())

    destination = ctx.resolve(path)
    if destination is None:
        raise BlockError("Invalid image-list path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ctx.logger.info("Wrote %d image path(s) to %s", len(lines), destination)
    return dataset


@block(
    "make_batches",
    "Export fixed-size dataset batches, optionally as ZIP archives.",
    aliases=("split_batches",),
)
def make_batches(
    dataset: Dataset,
    ctx: Context,
    *,
    path: str,
    batch_size: int = 100,
    split: str | None = None,
    default_split: str = "train",
    zip_archives: bool = True,
    overwrite: bool = False,
) -> Dataset:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise BlockError("batch_size must be a positive integer")
    if split is not None and split not in KNOWN_SPLITS:
        raise BlockError(f"split must be one of {', '.join(KNOWN_SPLITS)}, got {split!r}")

    records = dataset.images
    if split is not None:
        records = [record for record in records if (record.split or default_split) == split]
    if not records:
        raise BlockError("No images matched the requested split")

    destination = ctx.resolve(path)
    if destination is None:
        raise BlockError("Invalid batches path")
    _prepare_destination(destination, overwrite=overwrite)

    batch_count = (len(records) + batch_size - 1) // batch_size
    for offset in range(0, len(records), batch_size):
        batch_dir = destination / f"batch_{offset // batch_size + 1:03d}"
        batch_dataset = Dataset(
            images=records[offset : offset + batch_size],
            classes=dataset.classes,
            source_format=dataset.source_format,
            root=dataset.root,
            metadata=dict(dataset.metadata),
        )
        write_yolo_dataset(
            batch_dataset,
            batch_dir,
            default_split=default_split,
            mode="copy",
            overwrite=True,
        )
        if zip_archives:
            shutil.make_archive(str(batch_dir), "zip", root_dir=batch_dir)

    ctx.logger.info(
        "Created %d batch(es) of up to %d image(s) in %s",
        batch_count,
        batch_size,
        destination,
    )
    return dataset
