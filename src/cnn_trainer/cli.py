"""Command-line interface."""

from __future__ import annotations

import argparse
import inspect
import logging
import sys

from . import __version__
from . import blocks as _blocks  # noqa: F401  (registers the built-in blocks)
from .errors import CnnTrainerError
from .pipeline import run_pipeline
from .registry import alias_map, all_blocks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cnn-trainer",
        description="YAML-driven pre-processing and training pipelines for object detectors.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a pipeline YAML file")
    run_parser.add_argument("config", help="Path to the pipeline YAML file")
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load the dataset and show the block plan without executing it",
    )
    run_parser.add_argument(
        "--debug",
        action="store_true",
        help="Print the traceback when a pipeline fails",
    )

    blocks_parser = subparsers.add_parser("blocks", help="List available blocks")
    blocks_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show block parameters and defaults",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("cnn_trainer")

    if args.command == "blocks":
        _print_blocks(verbose=args.verbose)
        return 0

    try:
        manifest = run_pipeline(args.config, dry_run=args.dry_run, logger=logger)
    except CnnTrainerError as error:
        logger.error("error: %s", error)
        if args.debug:
            logger.exception("")
        return 1
    except Exception as error:  # pragma: no cover - safety net for the CLI
        logger.error("unexpected error: %s", error)
        if args.debug:
            logger.exception("")
        return 1

    if args.dry_run:
        print(
            f"Dry run OK: {manifest['dataset']['images']} images, "
            f"{manifest['dataset']['annotations']} annotations, "
            f"{len(manifest['classes'])} classes"
        )
    return 0


def _print_blocks(*, verbose: bool) -> None:
    aliases = alias_map()
    reverse_aliases: dict[str, list[str]] = {}
    for alias, canonical in aliases.items():
        reverse_aliases.setdefault(canonical, []).append(alias)

    for spec in all_blocks():
        alias_text = ""
        if reverse_aliases.get(spec.name):
            alias_text = f"  (aliases: {', '.join(sorted(reverse_aliases[spec.name]))})"
        print(f"{spec.name}{alias_text}")
        print(f"  {spec.description}")

        if verbose:
            signature = inspect.signature(spec.function)
            for name, parameter in signature.parameters.items():
                if name in {"dataset", "ctx"}:
                    continue
                if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                    continue
                default = (
                    ""
                    if parameter.default is inspect.Parameter.empty
                    else f" = {parameter.default!r}"
                )
                print(f"    {name}{default}")
        print()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
