import threading
from abc import abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
from pandas import DataFrame

from fenris.core.component import Component

SOME_CONSTANT = 0


class SomeClass:
    pass


class SomeOtherClass:
    pass


def some_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    return func


class Base(Component):
    @property
    @abstractmethod
    def keep_decorator(self) -> str:
        pass

    @abstractmethod
    def horizontal_args(self, x: int, y: bool, z: float) -> None:
        pass

    @abstractmethod
    def vertical_args(
        self,
        x: int,
        y: bool,
        z: float,
    ) -> None:
        pass

    @abstractmethod
    def reference_import_in_params(self, df: DataFrame) -> None:
        pass

    @abstractmethod
    def reference_import_in_return(self) -> threading.Thread:
        pass

    @abstractmethod
    def reference_import_asname(self) -> np.ndarray:
        pass

    @some_decorator
    @abstractmethod
    def reference_local(
        self, some: SomeClass, const: int = SOME_CONSTANT
    ) -> SomeOtherClass:
        pass
