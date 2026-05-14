from abc import ABC
from dataclasses import dataclass
from typing import Any, final


@dataclass(frozen=True)
class Metadata:
    """Registry metadata attached to every Component subclass."""

    name: str
    group: str
    value: str
    module: str
    attr: str
    dist_name: str
    dist_version: str


class Component(ABC):
    """The base for all pluggable components."""

    metadata: Metadata

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Enforce reserved-attribute rules on every Component subclass.

        Raises
        ------
        TypeError
            If the subclass declares ``metadata`` or ``name``.
        """
        super().__init_subclass__(**kwargs)
        if "metadata" in cls.__dict__:
            raise TypeError(
                "The 'metadata' attr is reserved, and set dynamically by the relevant "
                "registry."
            )
        if "name" in cls.__dict__:
            raise TypeError(
                "The 'name' property is implemented in the base class and should "
                "not not be overridden."
            )

    @final
    @property
    def name(self) -> str:
        """Registry name of this component."""
        return self.__class__.metadata.name
