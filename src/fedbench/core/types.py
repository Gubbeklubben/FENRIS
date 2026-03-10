from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray

# Avoid importing torch at runtime
if TYPE_CHECKING:
    import torch


type Arrays = list[NDArray[Any]] | dict[str, torch.Tensor]
type Objects = dict[str, Any]
type Metrics = dict[str, int | float | list[int] | list[float]]
type Extras = dict[
    str,
    str
    | bool
    | int
    | float
    | bytes
    | list[str]
    | list[bool]
    | list[int]
    | list[float]
    | list[bytes],
]