from abc import ABC, abstractmethod


class Component(ABC):
    """The base for all pluggable components."""

    @property
    @abstractmethod
    def id(self) -> str:
        """String identifying this component."""
        pass
