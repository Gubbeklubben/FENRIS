from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from numpy.typing import NDArray

# Avoid importing torch at runtime
if TYPE_CHECKING:
    import torch


type ModelState = list[NDArray] | dict[str, torch.Tensor]
type ConfigDict = dict[
    str, str | bool | int | float | bytes | list[str] | list[bool] | list[int]
    | list[float] | list[bytes]] | None
type MetricsDict = dict[str, int | float | list[int] | list[float]] | None


class MLRuntime(Enum):
    NUMPY = "numpy"
    TORCH = "torch"


@dataclass(frozen=True)
class InitRequest:
    client_id: int
    config: ConfigDict


@dataclass(frozen=True)
class InitResponse:
    client_id: int
    statistics: dict[str, list[NDArray]] | None


@dataclass(frozen=True)
class TrainRequest:
    client_id: int
    model_state: ModelState
    config: ConfigDict


@dataclass(frozen=True)
class TrainResponse:
    client_id: int
    model_state: ModelState | None
    metrics: dict[str, float] | None
    num_examples: int


@dataclass(frozen=True)
class EvalRequest:
    client_id: int
    model_state: ModelState
    config: ConfigDict


@dataclass(frozen=True)
class EvalResponse:
    client_id: int
    metrics: dict[str, float] | None
