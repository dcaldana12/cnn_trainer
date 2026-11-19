"""Render paginated image contact sheets for redundancy review."""

from __future__ import annotations

import shutil
from functools import lru_cache
from pathlib import Path
from tempfile import mkdtemp
from textwrap import wrap

from ..context import Context
from ..dataset import ImageRecord

_ROWS_PER_SHEET = 10
_TILE_WIDTH = 480
_PREVIEW_HEIGHT = 320
_ROW_HEIGHT = 420


def write_review(ctx: Context, records: list[ImageRecord], report: dict) -> Path:
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    parent = ctx.ensure_workdir() / "redundancy-review"
    parent.mkdir(exist_ok=True)
    directory = Path(mkdtemp(prefix="scan-", dir=parent))
    by_path = {record.relative_path.as_posix(): record for record in records}
    duplicates = sorted(
        (item for item in report["candidates"] if item["reason"] == "duplicate"),
        key=lambda item: item["group"],
    )
    pairs = [
        (
            (item["relative_path"], f"Candidate | {item['group']} | Distance {item['similarity']}"),
            (item["kept"], f"Keeper | {item['group']}"),
        )
        for item in duplicates
    ]
    blanks = [
        (item["relative_path"], f"Low contrast | {item['contrast']}")
        for item in report["candidates"]
        if item["reason"] == "blank"
    ]
    blank_rows = [blanks[start : start + 2] for start in range(0, len(blanks), 2)]
    try:
        font = ImageFont.load_default(size=15)
    except TypeError:  # Pillow 10.0 does not accept a font size.
        font = ImageFont.load_default()

    @lru_cache(maxsize=8)
    def load_preview(path: str):
        with Image.open(by_path[path].path) as source:
            source.draft("RGB", (_TILE_WIDTH, _PREVIEW_HEIGHT))
            image = ImageOps.exif_transpose(source)
            image.thumbnail((_TILE_WIDTH - 24, _PREVIEW_HEIGHT - 12))
            return image.convert("RGB")

    def draw_tile(sheet, draw, x: int, y: int, path: str, caption: str) -> None:
        preview = load_preview(path)
        sheet.paste(
            preview,
            (x + (_TILE_WIDTH - preview.width) // 2, y + (_PREVIEW_HEIGHT - preview.height) // 2),
        )
        draw.text((x + 12, y + _PREVIEW_HEIGHT), caption, font=font, fill="#ffffff")
        filename = "\n".join(wrap(path, width=53, max_lines=3, placeholder="..."))
        draw.multiline_text((x + 12, y + _PREVIEW_HEIGHT + 24), filename, font=font, fill="#d2daf0")

    try:
        for name, rows in (("duplicates", pairs), ("blanks", blank_rows)):
            for start in range(0, len(rows), _ROWS_PER_SHEET):
                page = rows[start : start + _ROWS_PER_SHEET]
                sheet = Image.new("RGB", (2 * _TILE_WIDTH, len(page) * _ROW_HEIGHT), "#172032")
                draw = ImageDraw.Draw(sheet)
                for row, cells in enumerate(page):
                    y = row * _ROW_HEIGHT
                    draw.line((0, y, sheet.width, y), fill="#627089")
                    for column, (path, caption) in enumerate(cells):
                        draw_tile(sheet, draw, column * _TILE_WIDTH, y, path, caption)
                draw.line((_TILE_WIDTH, 0, _TILE_WIDTH, sheet.height), fill="#627089")
                sheet.save(directory / f"{name}-{start // _ROWS_PER_SHEET + 1:03d}.png")
    except Exception:
        shutil.rmtree(directory)
        raise

    return directory
