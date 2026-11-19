"""cnn-trainer: block-based dataset pipelines for object detectors."""

from __future__ import annotations

from .dataset import Annotation, Dataset, ImageRecord
from .errors import BlockError, CnnTrainerError, ConfigError, DatasetError
from .pipeline import run_pipeline

__all__ = [
    "Annotation",
    "BlockError",
    "CnnTrainerError",
    "ConfigError",
    "Dataset",
    "DatasetError",
    "ImageRecord",
    "run_pipeline",
]

__version__ = "0.1.0"
