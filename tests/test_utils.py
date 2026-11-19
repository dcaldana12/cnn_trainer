"""Utility tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from cnn_trainer.errors import DatasetError
from cnn_trainer.utils import read_image_size, resolve_path, safe_relative_path, strip_split_prefix
from tests.helpers import write_png


def test_read_png_size_without_pillow(tmp_path):
    image = tmp_path / "image.png"
    write_png(image, 123, 45)
    assert read_image_size(image) == (123, 45)


def test_safe_relative_path_rejects_parent_traversal():
    with pytest.raises(DatasetError, match="Unsafe relative path"):
        safe_relative_path("../secret.png")


def test_safe_relative_path_normalizes_windows_paths():
    assert safe_relative_path(r"C:\dataset\images\a.jpg") == Path("a.jpg")


def test_strip_split_prefix():
    known = {"train", "val", "test"}
    assert strip_split_prefix(Path("train/a.png"), known) == Path("a.png")
    assert strip_split_prefix(Path("val/sub/a.png"), known) == Path("sub/a.png")
    assert strip_split_prefix(Path("sub/a.png"), known) == Path("sub/a.png")


def test_resolve_path_expands_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("CNN_TRAINER_DATA", str(tmp_path / "data"))
    resolved = resolve_path("$CNN_TRAINER_DATA/images", tmp_path)
    assert resolved == (tmp_path / "data" / "images").resolve()
