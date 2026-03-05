from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray

# Avoid importing torch at runtime
if TYPE_CHECKING:
    import torch


type Arrays  = list[NDArray[Any]] | dict[str, torch.Tensor]
type Objects = dict[str, Any]
type Metrics = dict[str, int | float | list[int] | list[float]]
type Extras  = dict[str, str | bool | int | float | bytes
               | list[str] | list[bool] | list[int] | list[float] | list[bytes]]

@dataclass(frozen=True)  # Can not replace top level dicts once created
class Update:
    arrays: dict[str, Arrays] = field(default_factory=dict)
    objects: dict[str, Objects] = field(default_factory=dict)
    metrics: dict[str, Metrics] = field(default_factory=dict)
    extras: dict[str, Extras] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return (  # nofmt
                not self.arrays
                and not self.objects
                and not self.metrics
                and not self.extras)
