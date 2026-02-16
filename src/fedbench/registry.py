import importlib
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


class Registry[T]:
    def __init__(
            self,
            group: str,
            validator: Callable[[T], T] | None = None) -> None:

        def default_validator(value: T) -> T:
            return value

        self._group = group
        self._validator = validator or default_validator
        self._builtins: dict[str, str] = {}
        self._entry_points = entry_points(group=group)

    def __iter__(self) -> Iterator[Metadata]:
        for k, v in self._builtins.items():
            yield Metadata(name=k, locator=v, source="builtin")

        for e in self._entry_points:
            yield Metadata(name=e.name, locator=e.value, source="plugin")

    def add_builtin(self, name: str, locator: str) -> None:
        if not _is_valid_locator(locator):
            raise ValueError(f"Invalid locator '{locator}'")
        self._builtins[name] = locator

    @property
    def group(self) -> str:
        return self._group

    def load(self, name: str) -> T:
        value = self._load_builtin(name)
        if value is not None:
            return value

        value = self._load_plugin(name)
        if value is None:
            raise ValueError(f"No such entry: {name}.")

        return value

    def _load_builtin(self, name: str) -> T | None:
        try:
            locator = self._builtins[name]
        except KeyError:
            return None

        module_name, _, qualifier = locator.partition(":")
        module = importlib.import_module(module_name)

        value: Any = module
        for attr in qualifier.split("."):
            if not hasattr(value, attr):
                raise ValueError(f"Bad locator '{locator}' in {self}")
            value = getattr(value, attr)

        return self._validator(cast(T, value))

    def _load_plugin(self, name: str) -> T | None:
        try:
            entry_point = self._entry_points[name]
        except KeyError:
            return None

        value = entry_point.load()
        return self._validator(value)


def _is_valid_locator(locator: str) -> bool:
    module, _, qualifier = locator.partition(":")

    def valid(s: str) -> bool:
        return s.isidentifier() and not keyword.iskeyword(s)

    if not all(valid(m) for m in module.split(".")):
        return False

    if not all(valid(q) for q in qualifier.split(".")):
        return False

    return True

