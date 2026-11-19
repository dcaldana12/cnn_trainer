"""Dataset inspection blocks."""

from __future__ import annotations

import json

from ..context import Context
from ..dataset import Dataset
from ..registry import block


@block("stats", "Print and record dataset counts.", aliases=("summary",))
def stats(dataset: Dataset, ctx: Context) -> Dataset:
    data = dataset.stats()
    ctx.logger.info(
        "Dataset: %d images, %d annotations, %d classes; splits: %s",
        data["images"],
        data["annotations"],
        len(dataset.classes),
        data["splits"],
    )
    ctx.logger.info("Classes: %s", dataset.classes)

    ctx.ensure_workdir()
    output = ctx.workdir / "stats.json"
    output.write_text(
        json.dumps(
            {
                "classes": dataset.classes,
                "stats": data,
                "parse_errors": dataset.metadata.get("parse_errors", []),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    ctx.logger.info("Wrote %s", output)
    return dataset
