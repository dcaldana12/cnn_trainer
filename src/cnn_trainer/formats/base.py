"""Shared helpers for source adapters."""

from __future__ import annotations

from ..errors import DatasetError


def normalize_class_names(value) -> list[str] | None:
    """Accept a list of names or an ``{index: name}`` mapping."""
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        names = [str(item) for item in value]
    elif isinstance(value, dict):
        try:
            items = sorted(value.items(), key=lambda item: int(item[0]))
        except (TypeError, ValueError) as error:
            raise DatasetError("class_names keys must be integer class ids") from error
        names = []
        for expected, (key, name) in enumerate(items):
            if int(key) != expected:
                raise DatasetError(
                    "class_names ids must be contiguous and start at 0; "
                    f"expected {expected}, got {key}"
                )
            names.append(str(name))
    else:
        raise DatasetError("class_names must be a list or an index -> name mapping")

    if not names:
        raise DatasetError("class_names must not be empty")
    if len(set(names)) != len(names):
        raise DatasetError(f"class_names contains duplicates: {names}")
    return names
