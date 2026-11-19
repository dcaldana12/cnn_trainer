"""YAML pipeline loading and execution."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from . import blocks as _blocks  # noqa: F401  (registers the built-in blocks)
from .context import Context
from .dataset import Dataset
from .errors import BlockError, CnnTrainerError, ConfigError
from .formats import load_dataset
from .registry import bind_parameters, get_block
from .utils import resolve_path

_ALLOWED_TOP_LEVEL = {"run", "source", "pipeline"}
_ALLOWED_STEP_KEYS = {"block", "enabled"}


def run_pipeline(
    config_path: str | Path,
    *,
    dry_run: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Load *config_path*, run its blocks, and return the run manifest."""
    config = load_config(config_path)
    logger = logger or logging.getLogger("cnn_trainer")
    ctx = Context(
        workdir=config["workdir"],
        config_dir=config["config_dir"],
        seed=config["seed"],
        run_name=config["name"],
        logger=logger,
    )

    started_at = _now()
    started = time.perf_counter()
    manifest: dict[str, Any] = {
        "name": config["name"],
        "config": str(config["path"]),
        "seed": config["seed"],
        "workdir": str(config["workdir"]),
        "started_at": started_at,
        "status": "running",
        "source": config["source"],
        "pipeline": config["pipeline"],
        "blocks": [],
    }

    try:
        dataset = load_dataset(config["source"], config["config_dir"])
    except CnnTrainerError:
        raise
    except Exception as error:
        raise ConfigError(f"Could not load source dataset: {error}") from error

    logger.info(
        "Loaded %s dataset from %s: %d images, %d annotations, %d classes",
        dataset.source_format,
        dataset.root,
        len(dataset.images),
        sum(len(record.annotations) for record in dataset.images),
        len(dataset.classes),
    )

    prepared = _prepare_steps(config["pipeline"], dataset, ctx)

    if dry_run:
        logger.info("Dry run: no blocks will be executed")
        for index, _name, spec, params, enabled in prepared:
            suffix = "" if enabled else " (disabled)"
            logger.info(
                "  %d. %s%s %s",
                index,
                spec.name,
                suffix,
                json.dumps(params, default=str, ensure_ascii=False),
            )
        manifest.update(
            {
                "status": "dry-run",
                "finished_at": _now(),
                "duration_s": round(time.perf_counter() - started, 3),
                "classes": dataset.classes,
                "dataset": dataset.stats(),
                "blocks": [
                    {
                        "block": name,
                        "canonical": spec.name,
                        "params": params,
                        "enabled": enabled,
                    }
                    for _, name, spec, params, enabled in prepared
                ],
            }
        )
        return manifest

    ctx.ensure_workdir()

    for index, name, spec, params, enabled in prepared:
        record: dict[str, Any] = {
            "block": name,
            "canonical": spec.name,
            "params": params,
            "enabled": enabled,
        }
        manifest["blocks"].append(record)
        if not enabled:
            record["status"] = "skipped"
            logger.info("[%d/%d] %s (disabled)", index, len(prepared), spec.name)
            continue

        logger.info("[%d/%d] %s", index, len(prepared), spec.name)
        block_started = time.perf_counter()
        try:
            result = spec.function(dataset, ctx, **params)
            if result is not None:
                dataset = result
            record["status"] = "ok"
            record["duration_s"] = round(time.perf_counter() - block_started, 3)
        except Exception as error:
            record["status"] = "failed"
            record["duration_s"] = round(time.perf_counter() - block_started, 3)
            record["error"] = str(error)
            manifest["status"] = "failed"
            manifest["error"] = f"Block {name!r} failed: {error}"
            manifest["finished_at"] = _now()
            manifest["duration_s"] = round(time.perf_counter() - started, 3)
            manifest["classes"] = dataset.classes
            manifest["dataset"] = dataset.stats()
            _write_manifest(ctx.workdir, manifest, logger)
            raise BlockError(f"Block {name!r} failed: {error}") from error

    manifest["status"] = "success"
    manifest["finished_at"] = _now()
    manifest["duration_s"] = round(time.perf_counter() - started, 3)
    manifest["classes"] = dataset.classes
    manifest["dataset"] = dataset.stats()

    try:
        _write_dataset_summary(ctx.workdir, dataset)
    except OSError as error:
        logger.warning("Could not write dataset summary: %s", error)
    _write_manifest(ctx.workdir, manifest, logger)

    logger.info(
        "Pipeline finished in %.2fs: %d images, %d annotations; run file: %s",
        manifest["duration_s"],
        len(dataset.images),
        sum(len(record.annotations) for record in dataset.images),
        ctx.workdir / "run.json",
    )
    return manifest


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"Pipeline config not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigError(f"Could not parse YAML {config_path}: {error}") from error

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError("The pipeline file must contain a YAML mapping")

    unknown = sorted(set(raw) - _ALLOWED_TOP_LEVEL)
    if unknown:
        raise ConfigError(
            f"Unknown top-level key(s): {', '.join(unknown)}. "
            f"Allowed: {', '.join(sorted(_ALLOWED_TOP_LEVEL))}"
        )

    source = raw.get("source")
    if not isinstance(source, dict):
        raise ConfigError("The pipeline needs a 'source' mapping")

    pipeline = raw.get("pipeline")
    if not isinstance(pipeline, list):
        raise ConfigError("The pipeline needs a 'pipeline' list")

    run = raw.get("run") or {}
    if not isinstance(run, dict):
        raise ConfigError("The 'run' section must be a mapping")

    unknown_run = sorted(set(run) - {"name", "seed", "workdir"})
    if unknown_run:
        raise ConfigError(
            f"Unknown run option(s): {', '.join(unknown_run)}. Allowed: name, seed, workdir"
        )

    name = str(run.get("name") or config_path.stem).strip()
    if not name:
        raise ConfigError("run.name must not be empty")
    if "/" in name or "\\" in name:
        raise ConfigError("run.name must not contain path separators")

    seed = run.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ConfigError("run.seed must be an integer")

    config_dir = config_path.parent
    configured_workdir = run.get("workdir")
    workdir = (
        resolve_path(configured_workdir, config_dir)
        if configured_workdir is not None
        else (config_dir / "runs" / name).resolve()
    )
    if workdir is None:
        raise ConfigError("Invalid run.workdir")

    return {
        "path": config_path,
        "config_dir": config_dir,
        "raw": raw,
        "name": name,
        "seed": seed,
        "workdir": workdir,
        "source": source,
        "pipeline": pipeline,
    }


def _prepare_steps(pipeline: list, dataset: Dataset, ctx: Context):
    prepared = []
    for index, step in enumerate(pipeline, start=1):
        if not isinstance(step, dict):
            raise ConfigError(f"pipeline step {index} must be a mapping")
        name = step.get("block")
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"pipeline step {index} needs a 'block' name")
        name = name.strip()

        enabled = step.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ConfigError(f"pipeline step {index}: 'enabled' must be true or false")

        params = {key: value for key, value in step.items() if key not in _ALLOWED_STEP_KEYS}
        spec = get_block(name)
        bind_parameters(spec, params, dataset, ctx)
        prepared.append((index, name, spec, params, enabled))
    return prepared


def _write_manifest(workdir: Path, manifest: dict[str, Any], logger: logging.Logger) -> None:
    path = workdir / "run.json"
    try:
        path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        logger.warning("Could not write run manifest %s: %s", path, error)


def _write_dataset_summary(workdir: Path, dataset: Dataset) -> None:
    summary = {
        "source_format": dataset.source_format,
        "root": str(dataset.root),
        "classes": dataset.classes,
        "stats": dataset.stats(),
        "orphan_labels": [str(path) for path in dataset.orphans],
        "parse_errors": dataset.metadata.get("parse_errors", []),
    }
    (workdir / "dataset.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
