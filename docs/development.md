# Development

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
git clone git@github.com:dcaldana12/cnn_trainer.git
cd cnn_trainer
uv venv
uv sync --extra dev
```

`uv venv` creates a project-local `./.venv`, so dependencies stay out of the
system Python. This also installs the package in editable mode and adds pytest,
ruff, Pillow, and coverage tooling. Activation is not required because
`uv run` uses the environment automatically.

If you only need the runtime blocks:

```bash
uv venv
uv sync                    # core: PyYAML
uv sync --extra images     # + Pillow image analysis
uv sync --extra train      # + Ultralytics training
uv sync --all-extras       # everything
```

## Common commands

```bash
uv run cnn-trainer blocks -v
uv run cnn-trainer run examples/pipeline.yaml --dry-run

uv run pytest
uv run pytest -q tests/test_pipeline.py
uv run ruff check src tests
uv run ruff format --check src tests
uv build
```

`uv run` uses the project environment, so no manual activation is required.

## Project layout

```text
src/cnn_trainer/
  __init__.py          Public package surface and version
  cli.py               argparse CLI: run, blocks
  pipeline.py          YAML loading, block execution, manifests
  registry.py          Block registration, aliases, parameter binding
  context.py           Per-run paths, seed, logger
  dataset.py           Dataset, ImageRecord, Annotation
  utils.py             Path and image helpers
  formats/             cvat.py, coco.py, yolo.py
  blocks/              One module per group of blocks
tests/                 Synthetic datasets and unit/integration tests
examples/              Runnable pipeline configurations
docs/                  Extended documentation
```

## Adding a block

1. Create a function in the relevant `src/cnn_trainer/blocks/*.py` module.

   ```python
   @block("my_block", "One-line description.", aliases=("legacy_name",))
   def my_block(
       dataset: Dataset,
       ctx: Context,
       *,
       threshold: float = 0.5,
   ) -> Dataset:
       ...
       return dataset
   ```

2. Register the module in `src/cnn_trainer/blocks/__init__.py`.
3. Validate parameters early and raise `BlockError` with an actionable message.
4. Keep the block deterministic; use `ctx.seed` for randomness.
5. Never write source files. Modify the in-memory dataset and let export blocks
   materialize output.
6. Add tests. Use the synthetic fixtures in `tests/conftest.py` or build a small
   dataset in the test.
7. Document the block in `docs/blocks.md` and, if it changes the common workflow,
   in the README quick start.

Block signatures are inspected at runtime. Variadic keyword arguments disable
parameter validation, so prefer explicit keyword parameters.

## Adding a source format

1. Implement `load(source: dict, root: Path, class_names) -> Dataset` in
   `src/cnn_trainer/formats/`.
2. Register the format and its allowed keys in `src/cnn_trainer/formats/__init__.py`.
3. Normalize boxes to absolute `xyxy`, map class ids densely, and populate
   `relative_path`, `width`, `height`, `has_label`, and `split` when available.
4. Add a fixture and loader tests, including a malformed-input case.
5. Document it in `docs/sources.md`.

## Code style

- Ruff handles linting and import order (`line-length = 100`, target Python 3.10).
- Prefer explicit types on public functions.
- Keep user-facing errors short and include the fix.
- Do not add mandatory heavy dependencies. Use an extra and raise a clear
  `BlockError` when the extra is missing.

## Tests

```bash
uv run pytest
uv run pytest --cov=cnn_trainer --cov-report=term-missing
```

The suite builds synthetic CVAT, COCO, and YOLO datasets and does not need
Ultralytics or a GPU. Image-analysis tests use Pillow from the dev extra.

When changing split allocation or export layout, add an end-to-end test in
`tests/test_pipeline.py` that inspects the generated `data.yaml` and files.

## Release checklist

1. Update `version` in `pyproject.toml` and `__version__` in
   `src/cnn_trainer/__init__.py`.
2. Update `CHANGELOG.md`.
3. Run the full suite and build:

   ```bash
   uv sync --all-extras
   uv run ruff check src tests
   uv run pytest
   uv build
   ```

4. Commit, tag (`git tag vX.Y.Z`), and push the tag.
5. Create a GitHub release from the tag.

Publishing to PyPI is not configured yet; the `uv build` artifacts are ready for a
future upload.

## Continuous integration

`.github/workflows/ci.yml` runs on pushes and pull requests:

- Python 3.10, 3.11, and 3.12;
- `uv sync --locked --extra dev`;
- Ruff lint;
- pytest;
- `uv build` on the default Python version.
