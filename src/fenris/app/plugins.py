from __future__ import annotations

import importlib.metadata
import inspect
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from importlib.metadata import entry_points
from typing import Iterator

from fenris.core.algorithm import Coordinator, Synthesizer
from fenris.core.component import Component, Metadata
from fenris.core.data import Partitioner
from fenris.core.eval import Evaluator

_ROOT_PKG = __name__.split(".")[0]


class Registry[T: Component]:
    def __init__(self, group: str, base: type[T]) -> None:
        self._group = group
        self._base = base
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
            yield self.get_metadata(ep.name)

    def get_metadata(self, name: str) -> Metadata:
        entry_point = self._get_entry_point(name)
        return Metadata(
            name=entry_point.name,
            group=entry_point.group,
            value=entry_point.value,
            module=entry_point.module,
            attr=entry_point.attr,
            dist_name=getattr(entry_point.dist, "name", ""),
            dist_version=getattr(entry_point.dist, "version", ""),
        )

    def load(self, name: str) -> type[T]:
        raw = self._get_entry_point(name).load()

        if not inspect.isclass(raw):
            raise TypeError(f"Object {raw} is not a class.")

        if not issubclass(raw, self._base):
            raise TypeError(f"Class {raw} is not a subclass of {self._base}.")

        if inspect.isabstract(raw):
            raise TypeError(f"Class {raw} is abstract.")

        cls: type[T] = raw
        cls.metadata = self.get_metadata(name)
        return cls

    def _get_entry_point(self, name: str) -> importlib.metadata.EntryPoint:
        try:
            return self._entry_points[name]
        except KeyError:
            raise KeyError(f"No registry entry for component named '{name}'") from None


class Group[T: Component]:
    def __init__(self, name: str, base: type[T]) -> None:
        self._name = name
        self._base = base
        self._entry_point_group = f"{_ROOT_PKG}.{self._name}"
        self._registry: Registry[T] | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def base(self) -> type[T]:
        return self._base

    @property
    def bases(self) -> Iterable[type[T]]:
        yield self._base
        for cls in self._base.__subclasses__():
            if inspect.isabstract(cls):
                # noinspection PyTypeChecker
                yield cls

    @property
    def entry_point_group(self) -> str:
        return self._entry_point_group

    @property
    def registry(self) -> Registry[T]:
        if self._registry is None:
            self._registry = Registry[T](self._entry_point_group, self._base)
        # noinspection PyTypeChecker
        return self._registry


@dataclass(frozen=True)
class _Plugins:
    synthesizers: Group[Synthesizer] = Group("synthesizers", Synthesizer)  # type: ignore[type-abstract]
    coordinators: Group[Coordinator] = Group("coordinators", Coordinator)  # type: ignore[type-abstract]
    partitioners: Group[Partitioner] = Group("partitioners", Partitioner)  # type: ignore[type-abstract]
    evaluators: Group[Evaluator] = Group("evaluators", Evaluator)  # type: ignore[type-abstract]

    @property
    def groups(self) -> Mapping[str, Group[Component]]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


plugins = _Plugins()
