from abc import ABC
from typing import Any, Callable

from fedbench.common import MLRuntime
from fedbench.errors import DuplicateComponentError

# python >= 3.12
type ComponentResolver[T] = Callable[[PluginRegistry], T]


class PluginRegistry(ABC):
    def __init__(self, ml_runtime: MLRuntime):
        if not isinstance(ml_runtime, MLRuntime):
            raise TypeError(f"Invalid ml_runtime '{ml_runtime}'")
        self._ml_runtime = ml_runtime
        self._plugins = {}

    def __repr__(self):
        return f"{self.__class__.__name__}({self._ml_runtime})"

    def _register(self, key: str, plugin: Callable[..., Any]):
        if key in self._plugins:
            raise DuplicateComponentError(f"{self}: {key} already registered")
        self._plugins[key] = plugin
        return plugin

    def _get(self, key: str) -> Callable[..., Any]:
        return self._plugins[key]

    def resolve_components[T](self, resolver: ComponentResolver[T]) -> T:
        return resolver(self)