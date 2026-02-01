from abc import ABC
from collections.abc import Callable
from typing import Any, Self

from fedbench.common import MLRuntime


class BaseRegistry(ABC):
    def __init__(self, ml_runtime: MLRuntime):
        if not isinstance(ml_runtime, MLRuntime):
            raise TypeError(f"Invalid ml_runtime '{ml_runtime}'")
        self._ml_runtime = ml_runtime

    def __repr__(self):
        return f"{self.__class__.__name__}({self._ml_runtime})"

    def _register(
            self, decorator_name: str, attr_name: str,
            plugin: Callable[..., Any]) -> Callable[..., Any]:

        attr = getattr(self, attr_name, None)
        if attr is not None:
            raise ValueError(f"{self}: {decorator_name} already registered")

        setattr(self, attr_name, plugin)
        return plugin

    def resolve_components[T](self, resolver: Callable[[Self], T]) -> T:
        return resolver(self)


# Split hierarchy to simplify early validation of registry types.
# The most straight forward approach I could think of.
class ClientRegistry(BaseRegistry, ABC):
    pass


class ServerRegistry(BaseRegistry, ABC):
    pass