"""Convert dataset images to three-channel grayscale copies."""

from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import mkdtemp

from ..context import Context
from ..dataset import KNOWN_SPLITS, Dataset
from ..errors import BlockError
from ..registry import block


@block("grayscale", "Convert dataset images to grayscale with three RGB channels.")
def grayscale(dataset: Dataset, ctx: Context) -> Dataset:
    """Write converted copies and switch the dataset to them after every image succeeds."""
    if not dataset.images:
        raise BlockError("Cannot convert an empty dataset to grayscale")
    try:
        from PIL import Image
    except ImportError as error:
        raise BlockError(
            "Grayscale conversion requires Pillow. Install the images extra."
        ) from error

    directory = Path(mkdtemp(prefix="grayscale-", dir=ctx.ensure_workdir()))
    converted = []
    try:
        for record in dataset.images:
            relative = record.relative_path
            if relative.is_absolute() or not relative.name or ".." in relative.parts:
                raise BlockError(f"Invalid relative image path: {relative}")
            split = record.split if record.split in KNOWN_SPLITS else "unassigned"
            target = directory / split / relative
            if target.exists():
                raise BlockError(f"Grayscale output path collision: {split}/{relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with Image.open(record.path) as image:
                    gray = image.convert("RGB").convert("L")
                    Image.merge("RGB", (gray, gray, gray)).save(target)
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                raise BlockError(f"Cannot convert {relative} to grayscale: {error}") from error
            converted.append(target)
    except Exception:
        shutil.rmtree(directory)
        raise

    for record, target in zip(dataset.images, converted, strict=True):
        record.path = target
    ctx.logger.info(
        "Converted %d images to three-channel grayscale in %s", len(converted), directory
    )
    return dataset
