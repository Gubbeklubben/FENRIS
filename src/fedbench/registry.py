import importlib
import keyword
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Iterator, Literal


@dataclass(frozen=True)
class Metadata:
    name: str
    locator: str
    source: Literal["builtin", "plugin"]


class Registry[T]:
    def __init__(self, group: str, validator: Callable[[T], T]) -> None:
        self._group = group
        self._validator = validator
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

        module_name, _, attr = locator.partition(":")
        module = importlib.import_module(module_name)

        if not hasattr(module, attr):
            raise ValueError(
                f"Bad locator '{locator}' in builtin algorithm registry")

        value = getattr(module, attr)
        return self._validator(value)

    def _load_plugin(self, name: str) -> T | None:
        try:
            entry_point = self._entry_points[name]
        except KeyError:
            return None

        value = entry_point.load()
        return self._validator(value)


def _is_valid_locator(locator: str) -> bool:
    module, _, attr = locator.partition(":")
    def valid(s: str) -> bool:
        return s.isidentifier() and not keyword.iskeyword(s)
    return all(valid(m) for m in module.split(".")) and valid(attr)
