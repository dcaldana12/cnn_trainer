"""Combine same-format datasets into a single class-ID space."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

from ..context import Context
from ..dataset import KNOWN_SPLITS, Dataset
from ..errors import BlockError, CnnTrainerError
from ..formats import load_dataset
from ..registry import block


@block("merge_datasets", "Merge datasets of the same format and remap class IDs.")
def merge_datasets(dataset: Dataset, ctx: Context, *, paths: list[str | dict]) -> Dataset:
    """Merge the current dataset and additional sources in the specified order."""
    if not isinstance(paths, list) or not paths:
        raise BlockError("merge_datasets.paths must be a nonempty list of roots or source mappings")

    sources = [dataset]
    for item in paths:
        options = {"root": item} if isinstance(item, str) else item
        if not isinstance(options, dict) or not isinstance(options.get("root"), str):
            raise BlockError("Each merge input needs a dataset root path")
        if not options["root"].strip():
            raise BlockError("Each merge input needs a nonempty dataset root path")
        if "format" in options:
            raise BlockError("Merge inputs inherit the pipeline source format")
        try:
            sources.append(
                load_dataset({**options, "format": dataset.source_format}, ctx.config_dir)
            )
        except CnnTrainerError as error:
            raise BlockError(f"Cannot merge dataset {options['root']!r}: {error}") from error

    occurrences = Counter(name for source in sources for name in dict.fromkeys(source.classes))
    classes = sorted(occurrences, key=lambda name: -occurrences[name])
    class_ids = {name: index for index, name in enumerate(classes)}
    records = []
    parse_errors = []
    for index, source in enumerate(sources, start=1):
        prefix = Path(f"dataset-{index:02d}")
        ids = {old: class_ids[name] for old, name in enumerate(source.classes)}
        for record in source.images:
            relative = record.relative_path
            if record.split in KNOWN_SPLITS:
                if len(relative.parts) > 1 and relative.parts[0] == record.split:
                    relative = Path(*relative.parts[1:])
                relative = Path(record.split) / prefix / relative
            else:
                relative = prefix / relative

            annotations = []
            for annotation in record.annotations:
                if annotation.class_id not in ids:
                    raise BlockError(
                        f"{source.root}/{record.relative_path}: unknown class ID "
                        f"{annotation.class_id}"
                    )
                annotations.append(replace(annotation, class_id=ids[annotation.class_id]))
            records.append(replace(record, relative_path=relative, annotations=annotations))

        parse_errors.extend(
            f"{prefix} ({source.root}): {error}"
            for error in source.metadata.get("parse_errors", [])
        )

    ctx.logger.info(
        "Merged %d %s datasets: %d images, %d classes",
        len(sources),
        dataset.source_format,
        len(records),
        len(classes),
    )
    return replace(
        dataset,
        images=records,
        classes=classes,
        orphans=[path for source in sources for path in source.orphans],
        metadata={
            "parse_errors": parse_errors,
            "merged_sources": [str(source.root) for source in sources],
        },
    )
