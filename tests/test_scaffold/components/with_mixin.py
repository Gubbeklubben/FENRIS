from abc import ABC, abstractmethod

from tests.test_scaffold.components.base import Base


class Mixin(ABC):
    @abstractmethod
    def mix_it_up(self) -> None:
        pass


class WithMixin(Base, Mixin, ABC):
    pass
