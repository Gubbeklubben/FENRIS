from __future__ import annotations

import importlib
from dataclasses import dataclass
from enum import Enum
from importlib.metadata import entry_points
from typing import Any, Iterator, Self

from fedbench.core.algorithm import Coordinator, Synthesizer
from fedbench.core.component import Component
from fedbench.core.data import Partitioner
from fedbench.core.eval import Evaluator

_ROOT_PKG = __package__.split(".")[0]


@dataclass(frozen=True)
class Metadata:
    name: str
    locator: str


# PyCharm static analysis is most pleased if using Enum not StrEnum,
# typer is happy as long as _value_ is str.
class Group(Enum):
    SYNTHESIZERS = ("synthesizers", Synthesizer)
    COORDINATORS = ("coordinators", Coordinator)
    PARTITIONERS = ("partitioners", Partitioner)
    EVALUATORS = ("evaluators", Evaluator)

    def __new__(cls, name: str, base: type[Component]) -> Self:
        obj = object.__new__(cls)
        obj._value_ = name
        return obj

    def __init__(self, _: str, base: type[Component]) -> None:
        self._base = base

    @property
    def base(self) -> type[Component]:
        return self._base

    @property
    def entry_point(self) -> str:
        return f"{_ROOT_PKG}.{self.value}"

    def get_registry(self) -> Registry:
        return Registry.get(f"{self.entry_point}")


class Registry:
    _instances: dict[str, Self] = {}

    @classmethod
    def get(cls, group: str) -> Self:
        try:
            return cls._instances[group]
        except KeyError:
            registry = cls(group)
            cls._instances[group] = registry
            return registry

    def __init__(self, group: str) -> None:
        self._group = group
        self._entry_points: dict[str, importlib.metadata.EntryPoint] = {}

        for ep in entry_points(group=group):
            if ep.name in self._entry_points:
                raise ValueError(
                    f"Duplicate component {ep.name} from {ep.value}. "
                    f"{self} already contains {ep.name} from "
                    f"{self._entry_points[ep.name].value}"
                )
            else:
                self._entry_points[ep.name] = ep

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(group={self._group})"

    def __iter__(self) -> Iterator[str]:
        yield from self._entry_points.keys()

    def __contains__(self, name: str) -> bool:
        return name in self._entry_points

    def metadata(self) -> Iterator[Metadata]:
        for ep in self._entry_points.values():
            yield Metadata(name=ep.name, locator=ep.value)

    def load(self, name: str) -> Any:
        try:
            entry_point = self._entry_points[name]
        except KeyError:
            raise ValueError(
                f"No registry entry for component named '{name}'"
            ) from None

        return entry_point.load()
