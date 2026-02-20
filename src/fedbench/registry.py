import importlib
import inspect
import keyword
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any, Iterator, Literal, cast


@dataclass(frozen=True)
class Metadata:
    name: str
    locator: str
    source: Literal["builtin", "plugin"]


class FactoryRegistry[T]:
    def __init__(self, group: str, product_cls: type[Any]) -> None:
        self._group = group
        self._product_cls = product_cls
        self._builtins: dict[str, str] = {}
        self._entry_points = entry_points(group=group)

    def __iter__(self) -> Iterator[Metadata]:
        for k, v in self._builtins.items():
            yield Metadata(name=k, locator=v, source="builtin")

        for e in self._entry_points:
            yield Metadata(name=e.name, locator=e.value, source="plugin")

    @property
    def group(self) -> str:
        return self._group

    def has_entry(self, name: str) -> bool:
        return name in self._builtins or name in self._entry_points.names

    def add_builtin(self, name: str, locator: str) -> None:
        if not _is_valid_locator(locator):
            raise ValueError(f"Invalid locator '{locator}'")
        self._builtins[name] = locator

    def call(self, name: str, **kwargs: Any) -> T:
        factory = self._load(name)
        instance = factory(**kwargs)

        if not isinstance(instance, self._product_cls):
            raise TypeError(
                f"Unexpected type {type(instance)} produced by factory {name}"
            )
        # noinspection PyUnnecessaryCast
        return cast(T, instance)

    def _load(self, name: str) -> Callable[..., Any]:
        factory = self._load_builtin(name)
        if factory is None:
            factory = self._load_plugin(name)

        if factory is None:
            raise ValueError(f"No such entry: {name}.")
        return factory

    def _load_builtin(self, name: str) -> Callable[..., Any] | None:
        try:
            locator = self._builtins[name]
        except KeyError:
            return None

        module_name, _, qualifier = locator.partition(":")
        module = importlib.import_module(module_name)

        factory = module
        if qualifier:
            for attr in qualifier.split("."):
                if not hasattr(factory, attr):
                    raise ValueError(f"Bad locator '{locator}' in {self}")
                factory = getattr(factory, attr)

        if inspect.ismodule(factory):
            raise ValueError(f"Module {factory} is not a valid factory.")

        return _check_callable(factory)

    def _load_plugin(self, name: str) -> Callable[..., Any] | None:
        try:
            entry_point = self._entry_points[name]
        except KeyError:
            return None

        factory = entry_point.load()
        return _check_callable(factory)


def _check_callable(factory: Callable[..., Any]) -> Callable[..., Any]:
    if not callable(factory):
        raise TypeError(f"{factory} is not callable.")
    return factory


def _is_valid_locator(locator: str) -> bool:
    module, _, qualifier = locator.partition(":")

    def valid(s: str) -> bool:
        return s.isidentifier() and not keyword.iskeyword(s)

    if not all(valid(m) for m in module.split(".")):
        return False

    if not qualifier:
        return True

    if not all(valid(q) for q in qualifier.split(".")):
        return False

    return True

