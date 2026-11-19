# Contributing

Thanks for taking the time to improve `cnn-trainer`. This project is small
enough that a clear issue and a focused pull request are usually enough.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to help

- Report a bug with a minimal pipeline and the relevant console output.
- Request a block or source format, with a short example of the input data.
- Improve the documentation or add an example pipeline.
- Add tests for an edge case: empty datasets, missing labels, odd paths, tiny
  boxes, duplicate stems, and similar real-world problems.

## Development setup

Requirements: Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/dcaldana12/cnn_trainer.git
cd cnn_trainer
uv venv
uv sync --all-extras

uv run cnn-trainer blocks -v
uv run pytest
uv run ruff check src tests
```

`uv venv` creates a project-local `./.venv`, keeping every dependency out of the
system Python. `uv run` uses that environment automatically.

To work without the optional training dependency:

```bash
uv venv
uv sync --extra dev
uv run pytest
```

## Guidelines

### Blocks

- Keep one block focused on one responsibility.
- Accept `dataset: Dataset` and `ctx: Context` as the first two positional
  parameters; all options are keyword-only parameters.
- Return the dataset, even if the block only writes a report.
- Validate parameters at the start and raise `BlockError` with the fix in the
  message.
- Never modify source files. Change the in-memory dataset and let export blocks
  write output.
- Use `ctx.seed` instead of a hidden global seed.
- Register the module in `src/cnn_trainer/blocks/__init__.py`.
- Add tests and document the block in `docs/blocks.md`.

Example:

```python
@block("drop_empty", "Drop images without annotations.", aliases=("remove_empty",))
def drop_empty(
    dataset: Dataset,
    ctx: Context,
    *,
    keep_background: bool = True,
) -> Dataset:
    if not keep_background:
        dataset.images = [record for record in dataset.images if record.annotations]
    return dataset
```

### Source formats

- Normalize boxes to absolute `xyxy`.
- Map class ids to a dense class list.
- Populate `relative_path`, `width`, `height`, `has_label`, and `split` when the
  format provides it.
- Reject unsafe paths and malformed label lines.
- Add a fixture and tests, including a malformed-input case.
- Document the format in `docs/sources.md`.

### Optional dependencies

The core install must stay lightweight. New heavy dependencies belong in an
extra in `pyproject.toml`, and the block must raise a clear `BlockError` when the
extra is missing. See `find_redundant_images` (Pillow) and `train` (Ultralytics).

## Tests

```bash
uv run pytest
uv run pytest --cov=cnn_trainer --cov-report=term-missing
```

Tests use synthetic CVAT, COCO, and YOLO datasets. They must not require a GPU or
network access. A test that depends on an extra should use
`pytest.importorskip` or a clear skip reason.

## Pull requests

- Keep the change focused; split unrelated refactors into separate PRs.
- Update tests and documentation in the same PR.
- Use clear commit messages; Conventional Commits are welcome
  (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`).
- Make sure CI is green:

```bash
uv run ruff check src tests
uv run pytest
uv build
```

- Describe the user-visible behavior and any new parameters in the PR template.

## Reporting bugs

Open an issue with:

- what you ran (the pipeline YAML with private paths removed);
- the console output and the failing block;
- the expected behavior;
- Python and package versions:

```bash
uv run python -c "import sys, cnn_trainer; print(sys.version); print(cnn_trainer.__version__)"
```

For security issues, follow [SECURITY.md](SECURITY.md) instead of opening a
public issue.
