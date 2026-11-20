"""YOLO training block."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from ..context import Context
from ..dataset import Dataset
from ..errors import BlockError
from ..registry import block
from ..utils import resolve_path
from .export import write_yolo_dataset


@block("train", "Train a YOLO detector with Ultralytics.")
def train(
    dataset: Dataset,
    ctx: Context,
    *,
    backend: str = "ultralytics",
    model: str | None = None,
    epochs: int | None = None,
    imgsz: int | list[int] | None = None,
    batch: int | float | None = None,
    device: str | int | list | None = None,
    workers: int | None = None,
    project: str | None = None,
    name: str | None = None,
    exist_ok: bool | None = None,
    seed: int | None = None,
    data: str | None = None,
    export_dir: str | None = None,
    extra: dict | None = None,
) -> Dataset:
    """Export the current dataset (unless ``data`` is given) and start training."""
    if backend != "ultralytics":
        raise BlockError(
            f"Unsupported training backend: {backend!r}. Only 'ultralytics' is available."
        )

    if extra is not None and not isinstance(extra, dict):
        raise BlockError("train.extra must be a mapping")

    train_args = {key: value for key, value in (extra or {}).items() if value is not None}
    train_args.update(
        (key, value)
        for key, value in {
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "device": device,
            "workers": workers,
            "project": project,
            "name": name,
            "exist_ok": exist_ok,
            "seed": seed,
        }.items()
        if value is not None
    )
    if "project" in train_args:
        train_args["project"] = str(ctx.resolve(train_args["project"]))

    config = {}
    config_dir = ctx.config_dir
    if "cfg" in train_args:
        cfg_path = ctx.resolve(train_args["cfg"])
        try:
            config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise BlockError(f"Cannot read Ultralytics training cfg {cfg_path}: {error}") from error
        if not isinstance(config, dict):
            raise BlockError(f"Ultralytics training cfg must contain a mapping: {cfg_path}")
        train_args["cfg"] = str(cfg_path)
        config_dir = cfg_path.parent

    extra_model = train_args.pop("model", None)
    model_dir = ctx.config_dir if model is not None or extra_model is not None else config_dir
    model = model or extra_model or config.get("model") or "yolov8n.pt"
    model = _local_path_or_backend_name(model, model_dir)

    configured_data = data if data is not None else train_args.get("data")
    if configured_data is not None:
        if data is not None:
            data_path = ctx.resolve(data)
            if not data_path.is_file():
                raise BlockError(f"Training data YAML not found: {data_path}")
            train_args["data"] = str(data_path)
        else:
            train_args["data"] = _resolve_backend_data(configured_data, ctx.config_dir)
    elif config.get("data"):
        train_args["data"] = _resolve_backend_data(config["data"], config_dir)
    else:
        splits = {record.split or "train" for record in dataset.images}
        if "train" not in splits:
            raise BlockError(
                "Training needs images in the train split. Add a split block before train."
            )
        if "val" not in splits:
            raise BlockError(
                "Ultralytics training needs a validation split. "
                "Add a split block with val > 0, or pass an existing data YAML via train.data."
            )

        destination = (
            ctx.resolve(export_dir) if export_dir else ctx.ensure_workdir() / "training_dataset"
        )
        if destination is None:
            raise BlockError("Invalid training export directory")
        data_path = write_yolo_dataset(
            dataset,
            destination,
            default_split="train",
            mode="copy",
            overwrite=True,
        )
        ctx.logger.info("Prepared YOLO training dataset at %s", destination)

        train_args["data"] = str(data_path)

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise BlockError(
            "Ultralytics is not installed. Install training support with: "
            "pip install 'cnn-trainer[train]'"
        ) from error

    ctx.logger.info("Starting Ultralytics training: model=%s, data=%s", model, train_args["data"])
    try:
        trainer_model = YOLO(model)
        # Ultralytics resolves an explicit path: . against the working directory.
        dataset_dir = _portable_dataset_dir(train_args["data"])
        original_dir = Path.cwd()
        if dataset_dir is not None:
            os.chdir(dataset_dir)
        try:
            trainer_model.train(**train_args)
            results_dir = Path(trainer_model.trainer.save_dir).resolve()
        finally:
            if dataset_dir is not None:
                os.chdir(original_dir)
    except Exception as error:  # Ultralytics raises a wide variety of exceptions.
        raise BlockError(f"Ultralytics training failed: {error}") from error

    ctx.logger.info("Training finished; results: %s", results_dir)
    dataset.metadata["training_results"] = str(results_dir)
    return dataset


def _portable_dataset_dir(value: str | list[str]) -> Path | None:
    if not isinstance(value, str):
        return None
    path = Path(value)
    if path.suffix not in {".yaml", ".yml"} or not path.is_file():
        return None
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        isinstance(config, dict)
        and config.get("path") == "."
        and config.get("train") == "train.txt"
    ):
        return path.resolve().parent
    return None


def _local_path_or_backend_name(value, directory):
    path = resolve_path(value, directory)
    return str(path) if path.is_file() else value


def _resolve_backend_data(value, directory):
    if isinstance(value, list):
        return [_local_path_or_backend_name(item, directory) for item in value]
    return _local_path_or_backend_name(value, directory)


__all__ = ["train"]
