"""Block registry.

Blocks are plain functions with the signature::

    def my_block(dataset: Dataset, ctx: Context, *, param=default) -> Dataset

The decorator registers the function under a canonical name and optional
aliases. Aliases exist so the original script names keep working.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .dataset import Dataset
from .errors import ConfigError

BlockFunction = Callable[..., Dataset | None]


@dataclass(frozen=True)
class BlockSpec:
    name: str
    function: BlockFunction
    description: str
    aliases: tuple[str, ...] = ()

    def parameter_names(self) -> list[str]:
        return [
            name
            for name, parameter in inspect.signature(self.function).parameters.items()
            if name not in {"dataset", "ctx"}
            and parameter.kind
            not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        ]

    def signature(self) -> inspect.Signature:
        return inspect.signature(self.function)


_REGISTRY: dict[str, BlockSpec] = {}
_ALIASES: dict[str, str] = {}


def block(name: str, description: str, aliases: tuple[str, ...] = ()):
    """Register a pipeline block."""

    def decorator(function: BlockFunction) -> BlockFunction:
        if name in _REGISTRY or name in _ALIASES:
            raise RuntimeError(f"Block name already registered: {name}")
        spec = BlockSpec(name=name, function=function, description=description, aliases=aliases)
        _REGISTRY[name] = spec
        for alias in aliases:
            if alias in _REGISTRY or alias in _ALIASES:
                raise RuntimeError(f"Block alias already registered: {alias}")
            _ALIASES[alias] = name
        return function

    return decorator


def get_block(name: str) -> BlockSpec:
    canonical = _ALIASES.get(name, name)
    try:
        return _REGISTRY[canonical]
    except KeyError as error:
        available = ", ".join(sorted(_REGISTRY))
        raise ConfigError(f"Unknown block {name!r}. Available blocks: {available}") from error


def all_blocks() -> list[BlockSpec]:
    return sorted(_REGISTRY.values(), key=lambda spec: spec.name)


def alias_map() -> dict[str, str]:
    return dict(_ALIASES)


def bind_parameters(spec: BlockSpec, params: dict[str, Any], dataset: Dataset, ctx) -> None:
    """Validate that *params* can be passed to the block."""
    signature = spec.signature()
    accepts_extra = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if not accepts_extra:
        allowed = set(spec.parameter_names())
        unknown = sorted(set(params) - allowed)
        if unknown:
            raise ConfigError(
                f"Block {spec.name!r} does not accept parameter(s): {', '.join(unknown)}. "
                f"Allowed: {', '.join(sorted(allowed)) or '(none)'}"
            )

    try:
        signature.bind(dataset, ctx, **params)
    except TypeError as error:
        raise ConfigError(f"Invalid parameters for block {spec.name!r}: {error}") from error
