from abc import ABC, abstractmethod

from tests.test_subclass_builder.components.base import Base


class OverrideAbstractWithAbstract(Base, ABC):
    @property
    @abstractmethod
    def keep_decorator(self) -> str:
        return ""
