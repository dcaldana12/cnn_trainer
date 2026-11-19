"""Per-run context handed to every block."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .utils import resolve_path


@dataclass
class Context:
    workdir: Path
    config_dir: Path
    seed: int
    run_name: str
    logger: logging.Logger

    def resolve(self, value: str | None) -> Path | None:
        """Resolve a path from the YAML relative to the config file."""
        return resolve_path(value, self.config_dir)

    def ensure_workdir(self) -> Path:
        self.workdir.mkdir(parents=True, exist_ok=True)
        return self.workdir
