from __future__ import annotations

import importlib
from dataclasses import dataclass
from enum import StrEnum
from importlib.metadata import entry_points
from typing import Any, Iterator, Self

_ROOT_PKG = __package__.split(".")[0]


@dataclass(frozen=True)
class Metadata:
    name: str
    locator: str


class Group(StrEnum):
    SYNTHESIZERS = "synthesizers"
    COORDINATORS = "coordinators"
    PARTITIONERS = "partitioners"
    EVALUATORS = "evaluators"

    def get_registry(self) -> Registry:
        return Registry.get(f"{_ROOT_PKG}.{self.value}")


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
