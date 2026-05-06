from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray

# Avoid importing torch at runtime
if TYPE_CHECKING:
    import torch


type Arrays = list[NDArray[Any]] | dict[str, torch.Tensor]
type Objects = dict[str, Any]
type Metrics = dict[
    str,
    int | float | list[int] | list[float],
]
type Extras = dict[
    str,
    str | bool | int | float | bytes |
    list[str] | list[bool] | list[int] | list[float] | list[bytes]
]  # fmt: skip


class ArraysTarget(StrEnum):
    """Deserialization target for the ``arrays`` field of a `Payload`.

    Determines whether array data is reconstructed as NumPy arrays or
    PyTorch tensors when a payload is decoded on the receiving side.
    """

    NUMPY = "numpy"
    TORCH = "torch"


@dataclass(frozen=True)
class Payload:
    """Container for data exchanged between the server and clients.

    Each field is a named dict so callers can store multiple independent
    datasets or parameter groups under distinct keys.

    Attributes
    ----------
    arrays : dict[str, Arrays]
        Named collections of numeric arrays (NumPy arrays or PyTorch tensors).
    objects : dict[str, Objects]
        Named JSON-serializable object dictionaries.
    metrics : dict[str, Metrics]
        Named metric dictionaries containing scalar or list numeric values.
    extras : dict[str, Extras]
        Named dictionaries for primitive scalar or list values.
    """

    arrays: dict[str, Arrays] = field(default_factory=dict)
    objects: dict[str, Objects] = field(default_factory=dict)
    metrics: dict[str, Metrics] = field(default_factory=dict)
    extras: dict[str, Extras] = field(default_factory=dict)

    def is_empty(self) -> bool:
        """Return ``True`` if all fields are empty.

        Returns
        -------
        bool
        """
        return (
            not self.arrays
            and not self.objects
            and not self.metrics
            and not self.extras
        )
