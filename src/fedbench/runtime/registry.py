"""
Plugin-aware factory registry.

Provides :class:`FactoryRegistry`, a generic registry that maps string
keys to callables (factories) and supports both built-in registrations
and third-party plugins discovered via Python package entry points.
"""

import importlib
import inspect
import keyword
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any, Iterator, Literal

from fedbench.core.logger import log_warning


@dataclass(frozen=True)
class Metadata:
    name: str
    locator: str
    source: Literal["builtin", "plugin"]


class FactoryRegistry[T]:
    def __init__(self, group: str, product_cls: type[T]) -> None:
        self._group = group

        if not isinstance(product_cls, type):
            raise TypeError("product_cls must be a type.")
        self._product_cls = product_cls

        self._builtins: dict[str, str] = {}
        self._plugins: dict[str, importlib.metadata.EntryPoint] = {}

        for ep in entry_points(group=group):
            if ep.name in self._plugins:
                log_warning(
                    str(self),
                    f"Ignoring duplicate plugin '{ep.name}' from '{ep.value}'",
                )
            else:
                self._plugins[ep.name] = ep

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(group={self.group}, "
            f"product_cls={self._product_cls})"
        )

    def __iter__(self) -> Iterator[str]:
        yield from self._builtins.keys()
        yield from self._plugins.keys()

    def __contains__(self, name: str) -> bool:
        return name in self._builtins or name in self._plugins

    @property
    def group(self) -> str:
        return self._group

    def add_builtin(self, name: str, locator: str) -> None:
        if not _is_valid_factory_locator(locator):
            raise ValueError(f"Invalid factory locator '{locator}'")

        if name in self._builtins:
            raise ValueError(f"Builtin factory '{name}' already registered.")

        self._builtins[name] = locator

    def call(self, name: str, factory_kwargs: dict[str, Any] | None = None) -> T:
        factory_kwargs = factory_kwargs or {}
        factory = self.load(name)

        if inspect.isclass(factory) and inspect.isabstract(factory):
            raise TypeError(f"{factory} is an abstract class.")

        if not callable(factory):
            raise TypeError(f"{factory} is not callable.")

        try:
            instance = factory(**factory_kwargs)
        except Exception as e:
            raise RuntimeError(f"Factory '{name}' raised an exception: {e}") from e
        if not isinstance(instance, self._product_cls):
            raise TypeError(
                f"Unexpected type {type(instance)} produced by factory {name}"
            )
        return instance

    def metadata(self) -> Iterator[Metadata]:
        for k, v in self._builtins.items():
            yield Metadata(name=k, locator=v, source="builtin")

        for ep in self._plugins.values():
            yield Metadata(name=ep.name, locator=ep.value, source="plugin")

    def load(self, name: str) -> Any:
        try:
            factory = self._load_builtin(name)
            if factory is None:
                factory = self._load_plugin(name)
        except (ValueError, TypeError):
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to load factory '{name}': {e}") from e

        if factory is None:
            raise ValueError(f"No such factory: '{name}'.")

        return factory

    def _load_builtin(self, name: str) -> Any | None:
        try:
            locator = self._builtins[name]
        except KeyError:
            return None

        module_name, _, qualifier = locator.partition(":")
        module = importlib.import_module(module_name)

        factory = module
        for attr in qualifier.split("."):
            if not hasattr(factory, attr):
                raise ValueError(f"Bad locator '{locator}' in {self}")
            factory = getattr(factory, attr)

        return factory

    def _load_plugin(self, name: str) -> Any | None:
        try:
            entry_point = self._plugins[name]
        except KeyError:
            return None

        return entry_point.load()


def _is_valid_factory_locator(locator: str) -> bool:
    module, _, qualifier = locator.partition(":")

    def valid(s: str) -> bool:
        return s.isidentifier() and not keyword.iskeyword(s)

    if not all(valid(m) for m in module.split(".")):
        return False

    if not all(valid(q) for q in qualifier.split(".")):
        return False

    return True
