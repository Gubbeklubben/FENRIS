from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from numpy.typing import NDArray

# Avoid importing torch at runtime
if TYPE_CHECKING:
    import torch


type MLRuntimeWeights = list[NDArray] | dict[str, torch.Tensor]


class MLRuntime(Enum):
    NUMPY = "numpy"
    TORCH = "torch"


@dataclass(frozen=True)
class TrainPlan:
    node_id: int
    ml_runtime_weights: MLRuntimeWeights
    config: dict[str, bool | int | float | bytes ]


@dataclass(frozen=True)
class TrainResult:
    ml_runtime_weights: MLRuntimeWeights
    metrics: dict[str, float]
    num_examples: int


@dataclass(frozen=True)
class EvalPlan:
    node_id: int
    ml_runtime_weights: MLRuntimeWeights
    config: dict[str, bool | int | float | bytes ]


@dataclass(frozen=True)
class EvalResult:
    metrics: dict[str, float]
