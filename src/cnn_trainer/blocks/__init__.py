"""Block package.

Importing this package registers every built-in block.
"""

from . import (
    audit,
    export,
    grayscale,
    labels,
    merge,
    output,
    redundancy,
    segmentation,
    split,
    stats,
    train,
    validate,
    visual_split,
)

__all__ = [
    "audit",
    "export",
    "grayscale",
    "labels",
    "merge",
    "output",
    "redundancy",
    "segmentation",
    "split",
    "stats",
    "train",
    "validate",
    "visual_split",
]
