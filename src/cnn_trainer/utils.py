"""Small path and image helpers.

The image-size reader deliberately avoids Pillow so that dataset-only pipelines
work with the minimal install. Pillow is used as a fallback when it is present.
"""

from __future__ import annotations

import os
import re
import struct
from pathlib import Path

from .errors import DatasetError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

_JPEG_SOF_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


def resolve_path(value: str | os.PathLike[str] | None, base: Path) -> Path | None:
    """Expand ``~``/environment variables and resolve *value* against *base*."""
    if value is None:
        return None
    expanded = os.path.expandvars(str(value))
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def safe_relative_path(value: str | os.PathLike[str]) -> Path:
    """Return a relative path that cannot escape its parent directory.

    Dataset formats occasionally store absolute paths or Windows separators.
    Absolute names are reduced to their final component; ``..`` components are
    rejected.
    """
    raw = str(value).replace("\\", "/")
    if re.match(r"^[A-Za-z]:/", raw):
        raw = raw.rsplit("/", 1)[-1]
    path = Path(raw)
    if path.is_absolute():
        path = Path(path.name)
    parts = [part for part in path.parts if part not in ("", ".")]
    if any(part == ".." for part in parts):
        raise DatasetError(f"Unsafe relative path in dataset: {value!r}")
    if not parts:
        raise DatasetError(f"Empty path in dataset: {value!r}")
    return Path(*parts)


def strip_split_prefix(relative_path: Path, known_splits: set[str] | frozenset[str]) -> Path:
    """Remove a leading ``train/``, ``val/`` or ``test/`` component.

    This keeps re-exported datasets from growing ``images/val/train/...``
    nests when a YOLO source already contains split folders.
    """
    parts = relative_path.parts
    if len(parts) > 1 and parts[0] in known_splits:
        return Path(*parts[1:])
    return relative_path


def iter_image_files(directory: Path) -> list[Path]:
    """Return image files below *directory*, sorted for deterministic runs."""
    if not directory.is_dir():
        raise DatasetError(f"Images directory not found: {directory}")
    files = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files, key=lambda path: path.as_posix())


def read_image_size(path: Path) -> tuple[int, int]:
    """Read ``(width, height)`` without decoding the image when possible."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(32)
            if head.startswith(b"\x89PNG\r\n\x1a\n"):
                handle.seek(16)
                width, height = struct.unpack(">II", handle.read(8))
                return int(width), int(height)
            if head.startswith(b"\xff\xd8"):
                return _jpeg_size(handle)
            if head.startswith(b"BM"):
                handle.seek(18)
                width, height = struct.unpack("<ii", handle.read(8))
                return abs(width), abs(height)
            if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
                size = _webp_size(handle)
                if size is not None:
                    return size
    except (OSError, struct.error) as error:
        raise DatasetError(f"Could not read image header for {path}: {error}") from error

    try:
        from PIL import Image
    except ImportError as error:
        raise DatasetError(
            f"Cannot determine the size of {path.name}. "
            "Install Pillow with: pip install 'cnn-trainer[images]'"
        ) from error

    try:
        with Image.open(path) as image:
            width, height = image.size
            return int(width), int(height)
    except OSError as error:
        raise DatasetError(f"Could not read image {path}: {error}") from error


def _jpeg_size(handle) -> tuple[int, int]:
    handle.seek(2)
    while True:
        byte = handle.read(1)
        if not byte:
            raise DatasetError("Unexpected end of JPEG while reading dimensions")
        if byte != b"\xff":
            continue
        marker = handle.read(1)
        while marker == b"\xff":
            marker = handle.read(1)
        if not marker:
            raise DatasetError("Unexpected end of JPEG while reading dimensions")
        marker_value = marker[0]
        if marker_value in (0xD8, 0xD9) or 0xD0 <= marker_value <= 0xD7:
            continue
        length_bytes = handle.read(2)
        if len(length_bytes) != 2:
            raise DatasetError("Unexpected end of JPEG while reading dimensions")
        length = struct.unpack(">H", length_bytes)[0]
        if marker_value in _JPEG_SOF_MARKERS:
            data = handle.read(5)
            if len(data) != 5:
                raise DatasetError("Unexpected end of JPEG while reading dimensions")
            height, width = struct.unpack(">HH", data[1:5])
            return int(width), int(height)
        handle.seek(length - 2, os.SEEK_CUR)


def _webp_size(handle) -> tuple[int, int] | None:
    handle.seek(12)
    chunk = handle.read(4)
    if chunk == b"VP8X":
        handle.seek(24)
        width = int.from_bytes(handle.read(3), "little") + 1
        height = int.from_bytes(handle.read(3), "little") + 1
        return width, height
    if chunk == b"VP8 ":
        handle.seek(26)
        data = handle.read(4)
        if len(data) != 4:
            return None
        width = struct.unpack("<H", data[0:2])[0] & 0x3FFF
        height = struct.unpack("<H", data[2:4])[0] & 0x3FFF
        return width, height
    if chunk == b"VP8L":
        handle.seek(20)
        data = handle.read(5)
        if len(data) != 5 or data[0] != 0x2F:
            return None
        bits = int.from_bytes(data[1:5], "little")
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return width, height
    return None
