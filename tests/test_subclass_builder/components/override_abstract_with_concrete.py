from abc import ABC

from tests.test_subclass_builder.components.base import Base


class OverrideAbstractWithConcrete(Base, ABC):
    @property
    def keep_decorator(self) -> str:
        return "keep_decorator"
