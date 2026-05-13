from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Metadata:
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
        super().__init_subclass__(**kwargs)
        attr = "metadata"
        if attr in cls.__dict__:
            raise TypeError(
                f"The '{attr}' attr is reserved, and set dynamically by the relevant "
                f"registry."
            )

    @property
    def name(self) -> str:
        return self.__class__.metadata.name
