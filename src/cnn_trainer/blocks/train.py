"""YOLO training block."""

from __future__ import annotations

from ..context import Context
from ..dataset import Dataset
from ..errors import BlockError
from ..registry import block
from .export import write_yolo_dataset


@block("train", "Train a YOLO detector with Ultralytics.")
def train(
    dataset: Dataset,
    ctx: Context,
    *,
    backend: str = "ultralytics",
    model: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int | list[int] = 640,
    batch: int | float = 16,
    device: str | int | list | None = None,
    workers: int = 8,
    project: str | None = None,
    name: str = "train",
    exist_ok: bool = True,
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

    if data:
        data_path = ctx.resolve(data)
        if data_path is None or not data_path.is_file():
            raise BlockError(f"Training data YAML not found: {data_path}")
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

    project_path = ctx.resolve(project) if project else ctx.ensure_workdir() / "training"
    if project_path is None:
        raise BlockError("Invalid training project directory")
    project_path.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise BlockError(
            "Ultralytics is not installed. Install training support with: "
            "pip install 'cnn-trainer[train]'"
        ) from error

    if extra is not None and not isinstance(extra, dict):
        raise BlockError("train.extra must be a mapping")

    train_args = dict(extra or {})
    train_args.update(
        {
            "data": str(data_path),
            "epochs": int(epochs),
            "imgsz": imgsz,
            "batch": batch,
            "workers": int(workers),
            "project": str(project_path),
            "name": name,
            "exist_ok": bool(exist_ok),
            "seed": int(ctx.seed if seed is None else seed),
        }
    )
    if device is not None:
        train_args["device"] = device

    ctx.logger.info(
        "Starting Ultralytics training: model=%s, epochs=%s, imgsz=%s, data=%s",
        model,
        epochs,
        imgsz,
        data_path,
    )

    try:
        YOLO(model).train(**train_args)
    except Exception as error:  # Ultralytics raises a wide variety of exceptions.
        raise BlockError(f"Ultralytics training failed: {error}") from error

    results_dir = project_path / name
    ctx.logger.info("Training finished; results: %s", results_dir)
    dataset.metadata["training_results"] = str(results_dir)
    return dataset


__all__ = ["train"]
