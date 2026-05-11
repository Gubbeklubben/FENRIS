from abc import ABC, abstractmethod

from tests.test_scaffold.components.base import Base


class OverrideAbstractWithAbstract(Base, ABC):
    @property
    @abstractmethod
    def keep_decorator(self) -> str:
        return ""
