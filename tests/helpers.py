"""Shared test helpers."""

from __future__ import annotations

import logging
import struct
import zlib
from pathlib import Path

from cnn_trainer.context import Context
from cnn_trainer.dataset import Annotation, Dataset, ImageRecord


def write_png(
    path: Path,
    width: int,
    height: int,
    color: tuple[int, int, int] = (30, 60, 90),
) -> None:
    """Write a minimal solid-colour PNG without Pillow."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = b"\x00" + bytes(color) * width
    raw = row * height

    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(payload)


def make_records(
    count: int,
    *,
    num_classes: int = 3,
    width: int = 100,
    height: int = 100,
) -> Dataset:
    classes = [f"class_{index}" for index in range(num_classes)]
    records = []
    for index in range(count):
        class_id = index % num_classes
        records.append(
            ImageRecord(
                path=Path(f"/dataset/{index}.png"),
                relative_path=Path(f"{index}.png"),
                width=width,
                height=height,
                annotations=[Annotation(class_id, 10.0, 10.0, 30.0, 30.0)],
            )
        )
    return Dataset(
        images=records,
        classes=classes,
        source_format="test",
        root=Path("/dataset"),
    )


def make_context(tmp_path: Path, *, seed: int = 42) -> Context:
    return Context(
        workdir=tmp_path / "run",
        config_dir=tmp_path,
        seed=seed,
        run_name="test",
        logger=logging.getLogger("cnn_trainer.test"),
    )
