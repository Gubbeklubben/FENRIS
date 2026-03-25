import importlib
import inspect
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any, Iterator


@dataclass(frozen=True)
class Metadata:
    name: str
    locator: str


class FactoryRegistry[T]:
    def __init__(self, group: str, product_cls: type[T]) -> None:
        self._group = group

        if not isinstance(product_cls, type):
            raise TypeError("product_cls must be a type.")
        self._product_cls = product_cls

        self._plugins: dict[str, importlib.metadata.EntryPoint] = {}

        for ep in entry_points(group=group):
            if ep.name in self._plugins:
                raise ValueError(
                    f"Duplicate component {ep.name} from {ep.value}. "
                    f"{self} already contains {ep.name} from "
                    f"{self._plugins[ep.name].value}"
                )
            else:
                self._plugins[ep.name] = ep

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(group={self.group}, "
            f"product_cls={self._product_cls})"
        )

    def __iter__(self) -> Iterator[str]:
        yield from self._plugins.keys()

    def __contains__(self, name: str) -> bool:
        return name in self._plugins

    @property
    def group(self) -> str:
        return self._group

    def call(self, name: str, factory_kwargs: dict[str, Any] | None = None) -> T:
        factory_kwargs = factory_kwargs or {}
        factory = self.load(name)

        if inspect.isclass(factory) and inspect.isabstract(factory):
            raise TypeError(f"{factory} is an abstract class.")

        if not callable(factory):
            raise TypeError(f"{factory} is not callable.")

        instance = factory(**factory_kwargs)
        if not isinstance(instance, self._product_cls):
            raise TypeError(
                f"Unexpected type {type(instance)} produced by factory {name}"
            )
        return instance

    def metadata(self) -> Iterator[Metadata]:
        for ep in self._plugins.values():
            yield Metadata(name=ep.name, locator=ep.value)

    def load(self, name: str) -> Any:
        factory = self._load_plugin(name)

        if factory is None:
            raise ValueError(f"No such factory: '{name}'.")

        return factory

    def _load_plugin(self, name: str) -> Any | None:
        try:
            entry_point = self._plugins[name]
        except KeyError:
            return None

        return entry_point.load()
